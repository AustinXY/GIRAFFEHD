import os

import argparse
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm
from PIL import Image
import wandb
import pickle

from calc_inception import load_patched_inception_v3
from scipy import linalg

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
)

from dataset import LMDBDataset
from model import GIRAFFEHDGenerator, Discriminator
from op import conv2d_gradfix


@torch.no_grad()
def extract_feature_from_model(
    args, generator, inception
):
    batch_size = args.fid_batch
    n_sample = args.fid_n_sample
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch
    features = []

    for batch in tqdm(batch_sizes):
        img = generator(batch, inject_index=args.eval_inj_idx)[0]
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to('cpu'))

    features = torch.cat(features, 0)
    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace
    return fid


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def binarization_loss(mask):
    return torch.min(1-mask, mask).mean()


def save_img(img, fname, nrow=8, normalize=True, range=(-1,1)):
    utils.save_image(img,
        fname,
        nrow=nrow,
        normalize=normalize,
        range=range,
    )


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, inception):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter,
                    dynamic_ncols=True, smoothing=0.01)

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))

    generator.train()
    discriminator.train()
    g_ema.eval()

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')
            break

        bbox_pad = args.bbox_pad
        if i <= 10000:
            bbox_pad = args.bbox_pad * (i / 10000)

        real_img = next(loader)
        real_img = real_img.to(device)

        ############# train discriminator network #############
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # sample image representation:
        # [z_s_fg, z_a_fg, z_s_bg, z_a_bg, camera rotation, elevation, radius,
        #  scale, translation, object rotation]
        img_rep = g_module.get_rand_rep(args.batch)
        fake_img = generator(img_rep)[0]

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict['d'] = d_loss
        loss_dict['real_score'] = real_pred.mean()
        loss_dict['fake_score'] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every +
             0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict['r1'] = r1_loss

        ############# train generator network #############
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        img_rep = g_module.get_rand_rep(args.batch)
        img_li = generator(img_rep, return_ids=[0, 5])
        fake_img = img_li[0]
        mask = img_li[5]

        bbox = g_module.get_2Dbbox(img_rep, padd=bbox_pad)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        bbox_loss = torch.mean((1 - bbox) * mask)

        cvg_loss = F.relu(args.min_ratio - torch.mean(mask))

        bin_loss = binarization_loss(mask)

        loss = g_loss + bbox_loss * args.bbox + \
            cvg_loss * args.cvg + bin_loss * args.bin

        loss_dict['g'] = g_loss
        loss_dict['bbox'] = bbox_loss
        loss_dict['cvg'] = cvg_loss
        loss_dict['bin'] = bin_loss

        generator.zero_grad()
        loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        bbox_loss_val = loss_reduced['bbox'].mean().item()
        bin_loss_val = loss_reduced['bin'].mean().item()
        cvg_loss_val = loss_reduced['cvg'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        real_score_val = loss_reduced['real_score'].mean().item()
        fake_score_val = loss_reduced['fake_score'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; '
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        'Generator': g_loss_val,
                        'Discriminator': d_loss_val,
                        'Real Score': real_score_val,
                        'Fake Score': fake_score_val,
                        'BBox': bbox_loss_val,
                        'FG Coverage': cvg_loss_val,
                        'Binarization': bin_loss_val,
                    }
                )

            if i % args.visualize_every == 0:
                if args.calcfid:
                    # calculate fid
                    features = extract_feature_from_model(
                        args, g_ema, inception
                    ).numpy()
                    print(f'extracted {features.shape[0]} features')
                    sample_mean = np.mean(features, 0)
                    sample_cov = np.cov(features, rowvar=False)

                    with open(f'inception_{args.dataset}_{args.fid_n_sample}_{args.size}.pkl', 'rb') as f:
                        embeds = pickle.load(f)
                        real_mean = embeds['mean']
                        real_cov = embeds['cov']

                    fid = calc_fid(sample_mean, sample_cov,
                                   real_mean, real_cov)

                    if wandb and args.wandb:
                        wandb.log(
                            {
                                'FID': fid,
                            }
                        )

                with torch.no_grad():
                    g_ema.eval()

                    img_rep = g_ema.get_rand_rep(args.n_sample)

                    img_li = g_ema(img_rep, return_ids=[], mode='eval', inject_index=args.eval_inj_idx)
                    fnl_img, fg_img, bg_img, _fg_img, fg_residual, fg_mk = img_li

                    # change fg_shape
                    img_rep[0] = g_ema.get_rand_rep(args.n_sample)[0]
                    fnl_img1 = g_ema(img_rep, mode='eval', inject_index=args.eval_inj_idx)[0]

                    # change fg_app
                    img_rep[1] = g_ema.get_rand_rep(args.n_sample)[1]
                    fnl_img2 = g_ema(img_rep, mode='eval', inject_index=args.eval_inj_idx)[0]

                    save_img(fnl_img, f'sample/{str(i).zfill(6)}_0.png')
                    save_img(fnl_img1, f'sample/{str(i).zfill(6)}_1.png')
                    save_img(fnl_img2, f'sample/{str(i).zfill(6)}_2.png')
                    save_img(fg_mk, f'sample/{str(i).zfill(6)}_3.png', range=(0,1))
                    save_img(fg_img, f'sample/{str(i).zfill(6)}_4.png')
                    save_img(bg_img, f'sample/{str(i).zfill(6)}_5.png')
                    save_img(torch.tanh(_fg_img), f'sample/{str(i).zfill(6)}_6.png')
                    save_img(torch.tanh(fg_residual), f'sample/{str(i).zfill(6)}_7.png')

                    if wandb and args.wandb:
                        wandb.log(
                            {
                                'image': [wandb.Image(Image.open(f'sample/{str(i).zfill(6)}_0.png').convert('RGB'))],
                                'change shape': [wandb.Image(Image.open(f'sample/{str(i).zfill(6)}_1.png').convert('RGB'))],
                                'change app': [wandb.Image(Image.open(f'sample/{str(i).zfill(6)}_2.png').convert('RGB'))],
                                'mask': [wandb.Image(Image.open(f'sample/{str(i).zfill(6)}_3.png').convert('RGB'))],
                                'fg_img': [wandb.Image(Image.open(f'sample/{str(i).zfill(6)}_4.png').convert('RGB'))],
                                'bg_img': [wandb.Image(Image.open(f'sample/{str(i).zfill(6)}_5.png').convert('RGB'))],
                                '_fg_img': [wandb.Image(Image.open(f'sample/{str(i).zfill(6)}_6.png').convert('RGB'))],
                                'fg_residual': [wandb.Image(Image.open(f'sample/{str(i).zfill(6)}_7.png').convert('RGB'))],
                            }
                        )

            if i % args.checkpoint_every == 0 and i != args.start_iter:
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                        'args': args,
                        'cur_itr': i,
                    },
                    f'checkpoint/{str(i).zfill(6)}_{args.ckpt_suf}.pt',
                )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser(description='GiraffeHD trainer')

    # training settings
    parser.add_argument('path', type=str, help='path to the lmdb dataset')
    parser.add_argument('--ckpt', type=str, default=None, help='path to the checkpoints to resume training')

    parser.add_argument('--dataset', type=str, default='compcar', help='training dataset')
    parser.add_argument('--datasize', type=int, default=0, help='dataset lmdb image size')
    parser.add_argument('--calcfid', action='store_true', help='use calculate fid during training')

    parser.add_argument('--fid_batch', type=int, default=8, help='batch sizes for fid calculation')
    parser.add_argument('--fid_n_sample', type=int, default=10000, help='number of the samples generated for fid calculation')

    parser.add_argument('--n_sample', type=int, default=8, help='number of the samples generated during training')
    parser.add_argument('--eval_inj_idx', type=int, default=4, help='inject index for evaluation')
    parser.add_argument('--visualize_every', type=int, default=500, help='interval of visualize training results')
    parser.add_argument('--checkpoint_every', type=int, default=10000, help='interval of the saving checkpoints')

    parser.add_argument('--wandb', action='store_true', help='use weights and biases logging')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

    # model hyperparams
    parser.add_argument('--size', type=int, default=128, help='generated image resolution')
    parser.add_argument('--res_vol', type=int, default=16, help='volume render resolution')
    parser.add_argument('--feat_dim', type=int, default=256,
                        help='feature dimension of neural feature field')
    parser.add_argument('--z_dim', type=int, default=256, help='foreground z dimension')
    parser.add_argument('--z_dim_bg', type=int, default=256, help='background z dimension')
    parser.add_argument('--channel_multiplier', type=int, default=2,
                        help='channel multiplier factor for the model. config-f = 2, else = 1')

    # training hyperparams
    parser.add_argument('--bbox_pad', type=float, default=0.1, help='padding of 3D bounding box when project to 2D')
    parser.add_argument('--iter', type=int, default=1200000, help='total training iterations')
    parser.add_argument('--batch', type=int, default=16, help='batch sizes for each gpus')
    parser.add_argument('--bin', type=float, default=4., help='mask binarization loss weight')
    parser.add_argument('--bbox', type=float, default=10., help='bounding box containment loss weight')
    parser.add_argument('--cvg', type=float, default=10., help='foreground coverage loss weight')
    parser.add_argument('--min_ratio', type=float, default=0.2,
                        help='foreground coverage loss minimum coverage threshold')
    parser.add_argument('--mix_prob', type=float, default=0.9,
                        help='probability of latent code mixing during training')

    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--r1', type=float, default=10, help='r1 regularization weight')
    parser.add_argument('--d_reg_every', type=int, default=16, help='interval of the applying r1 regularization')

    args = parser.parse_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.calcfid:
        assert os.path.exists(
            f'inception_{args.dataset}_{args.fid_n_sample}_{args.size}.pkl')

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        synchronize()

    args.start_iter = 0
    args.ckpt_suf = f'_{args.dataset}_{args.size}'

    if args.datasize == 0:
        args.datasize = args.size

    args.fid = -1
    if args.ckpt is not None:
        print('load model:', args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        ckpt_args = ckpt['args']
        args.start_iter = ckpt['cur_itr']

    assert args.dataset in ['compcar', 'ffhq', 'afhq', 'cats', 'celeba']

    args.grf_use_mlp = True
    args.use_viewdirs = False
    args.grf_use_z_app = False
    args.bbox_n_steps = 64

    if args.dataset == 'compcar':
        # giraffe hyperparams
        args.scale_range_min = [0.2, 0.16, 0.16]
        args.scale_range_max = [0.25, 0.2, 0.2]
        args.translation_range_min = [-0.22, -0.12, -0.06]
        args.translation_range_max = [0.22, 0.12, 0.08]
        args.rotation_range = [0., 1.]
        args.bg_translation_range_min = [-0.2, -0.2, 0.]
        args.bg_translation_range_max = [0.2, 0.2, 0.]
        args.bg_rotation_range = [0., 0.]
        args.range_u = [0., 0.]
        args.range_v = [0.41667, 0.5]
        args.fov = 10
        args.pos_share = True
        args.fg_gen_mask = True
        # loss hyperparams
        args.cvg = 14
        args.bbox = 14
        args.bin = 4
        args.min_ratio = 0.3
        args.mix_prob = 0.5

    elif args.dataset == 'ffhq':
        # giraffe hyperparams
        args.scale_range_min = [0.21, 0.21, 0.21]
        args.scale_range_max = [0.21, 0.21, 0.21]
        args.translation_range_min = [0., 0., 0.]
        args.translation_range_max = [0., 0., 0.]
        args.rotation_range = [0.40278, 0.59722]
        args.bg_translation_range_min = [-0.2, -0.2, 0.]
        args.bg_translation_range_max = [0.2, 0.2, 0.]
        args.bg_rotation_range = [0., 0.]
        args.range_u = [0., 0.]
        args.range_v = [0.4167, 0.5]
        args.fov = 10
        args.pos_share = False
        args.fg_gen_mask = False
        # loss hyperparams
        args.cvg = 14
        args.bbox = 14
        args.bin = 4
        args.min_ratio = 0.3
        args.mix_prob = 0.8

    elif args.dataset == 'afhq':
        # giraffe hyperparams
        args.scale_range_min = [0.21, 0.21, 0.21]
        args.scale_range_max = [0.21, 0.21, 0.21]
        args.translation_range_min = [0., 0., 0.]
        args.translation_range_max = [0., 0., 0.]
        args.rotation_range = [0.40278, 0.59722]
        args.bg_translation_range_min = [-0.2, -0.2, 0.]
        args.bg_translation_range_max = [0.2, 0.2, 0.]
        args.bg_rotation_range = [0., 0.]
        args.range_u = [0., 0.]
        args.range_v = [0.4167, 0.5]
        args.fov = 10
        args.pos_share = False
        args.fg_gen_mask = False
        # loss hyperparams
        args.cvg = 14
        args.bbox = 12
        args.bin = 4
        args.min_ratio = 0.3
        args.mix_prob = 0.2

    elif args.dataset == 'cats':
        # giraffe hyperparams
        args.scale_range_min = [0.21, 0.21, 0.21]
        args.scale_range_max = [0.21, 0.21, 0.21]
        args.translation_range_min = [0., 0., 0.]
        args.translation_range_max = [0., 0., 0.]
        args.rotation_range = [0.40278, 0.59722]
        args.bg_translation_range_min = [-0.2, -0.2, 0.]
        args.bg_translation_range_max = [0.2, 0.2, 0.]
        args.bg_rotation_range = [0., 0.]
        args.range_u = [0., 0.]
        args.range_v = [0.4167, 0.5]
        args.fov = 10
        args.pos_share = False
        args.fg_gen_mask = False
        # loss hyperparams
        args.cvg = 14
        args.bbox = 14
        args.bin = 4
        args.min_ratio = 0.3
        args.mix_prob = 0.2

    elif args.dataset == 'celeba':
        # giraffe hyperparams
        args.scale_range_min = [0.21, 0.21, 0.21]
        args.scale_range_max = [0.21, 0.21, 0.21]
        args.translation_range_min = [0., 0., 0.]
        args.translation_range_max = [0., 0., 0.]
        args.rotation_range = [0.375, 0.625]
        args.bg_translation_range_min = [-0.2, -0.2, 0.]
        args.bg_translation_range_max = [0.2, 0.2, 0.]
        args.bg_rotation_range = [0., 0.]
        args.range_u = [0., 0.]
        args.range_v = [0.4167, 0.5]
        args.fov = 10
        args.pos_share = False
        args.fg_gen_mask = False
        # loss hyperparams
        args.cvg = 14
        args.bbox = 14
        args.bin = 4
        args.min_ratio = 0.3
        args.mix_prob = 0.8

    generator = GIRAFFEHDGenerator(
        device=device,
        z_dim=args.z_dim,
        z_dim_bg=args.z_dim_bg,
        size=args.size,
        resolution_vol=args.res_vol,
        feat_dim=args.feat_dim,
        range_u=args.range_u,
        range_v=args.range_v,
        fov=args.fov,
        scale_range_max=args.scale_range_max,
        scale_range_min=args.scale_range_min,
        translation_range_max=args.translation_range_max,
        translation_range_min=args.translation_range_min,
        rotation_range=args.rotation_range,
        bg_translation_range_max=args.bg_translation_range_max,
        bg_translation_range_min=args.bg_translation_range_min,
        bg_rotation_range=args.bg_rotation_range,
        refine_n_styledconv=2,
        refine_kernal_size=3,
        mix_prob=args.mix_prob,
        grf_use_mlp=args.grf_use_mlp,
        pos_share=args.pos_share,
        use_viewdirs=args.use_viewdirs,
        grf_use_z_app=args.grf_use_z_app,
        fg_gen_mask=args.fg_gen_mask
    ).to(device)

    discriminator = Discriminator(
        size=args.size,
        channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema = GIRAFFEHDGenerator(
        device=device,
        z_dim=args.z_dim,
        z_dim_bg=args.z_dim_bg,
        size=args.size,
        resolution_vol=args.res_vol,
        feat_dim=args.feat_dim,
        range_u=args.range_u,
        range_v=args.range_v,
        fov=args.fov,
        scale_range_max=args.scale_range_max,
        scale_range_min=args.scale_range_min,
        translation_range_max=args.translation_range_max,
        translation_range_min=args.translation_range_min,
        rotation_range=args.rotation_range,
        bg_translation_range_max=args.bg_translation_range_max,
        bg_translation_range_min=args.bg_translation_range_min,
        bg_rotation_range=args.bg_rotation_range,
        refine_n_styledconv=2,
        refine_kernal_size=3,
        mix_prob=args.mix_prob,
        grf_use_mlp=args.grf_use_mlp,
        pos_share=args.pos_share,
        use_viewdirs=args.use_viewdirs,
        grf_use_z_app=args.grf_use_z_app,
        fg_gen_mask=args.fg_gen_mask
    ).to(device)

    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    inception = None
    if args.calcfid:
        inception = nn.DataParallel(load_patched_inception_v3()).to(device)
        inception.eval()

    if args.ckpt is not None:
        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])

        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    if args.datasize == args.size:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.transforms.Resize(args.size, Image.LANCZOS),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

    dataset = LMDBDataset(args.path, transform, args.datasize)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True,
                             distributed=args.distributed),
        drop_last=True,
    )

    print(args, '\n')

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project='giraffeHD')

    train(args, loader, generator, discriminator,
          g_optim, d_optim, g_ema, device, inception)
