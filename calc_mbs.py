# using s code encoder to repredict s code for color, shape disentanglement

import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from torch import nn
from tqdm import tqdm

from model import GIRAFFEHDGenerator
from torchvision.models.segmentation import deeplabv3_resnet101
from scipy import linalg
from torch.nn import functional as F
from torchvision import transforms, utils


def get_mask(model, batch, cid):
    normalized_batch = transforms.functional.normalize(
        batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    output = model(normalized_batch)['out']
    # sem_classes = [
    #     '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    #     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    # ]
    # sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    # cid = sem_class_to_idx['car']

    normalized_masks = torch.nn.functional.softmax(output, dim=1)

    boolean_car_masks = (normalized_masks.argmax(1) == cid)
    return boolean_car_masks.float()


def latent_change_fg(generator, latents):
    # random sample z_s_fg, z_a_fg, transformations
    batch = latents[0].size(0)
    latents_ = generator.get_rand_rep(batch)
    # [z_s_fg, z_a_fg, s, t, rval]
    for i in [0,1,7,9]:
        latents[i] = latents_[i]
    latents[8][:,0] = latents_[8][:,0]
    latents[8][:,1] = latents_[8][:,1]

    return latents


def norm_ip(img, low, high):
    img_ = img.clamp(min=low, max=high)
    img_.sub_(low).div_(max(high - low, 1e-5))
    return img_


def norm_range(t, value_range=(-1, 1)):
    if value_range is not None:
        return norm_ip(t, value_range[0], value_range[1])
    else:
        return norm_ip(t, float(t.min()), float(t.max()))

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser(description='Giraffe trainer')

    parser.add_argument(
        '--batch', type=int, default=16, help='batch sizes for each gpus'
    )
    parser.add_argument(
        '--n_sample',
        type=int,
        default=1000,
        help='number of the samples generated during training',
    )

    parser.add_argument(
        '--ckpt',
        type=str,
        default=None,
        help='path to the checkpoints to resume training',
    )
    parser.add_argument(
        '--channel_multiplier',
        type=int,
        default=2,
        help='channel multiplier factor for the model. config-f = 2, else = 1',
    )
    parser.add_argument(
        '--local_rank', type=int, default=0, help='local rank for distributed training'
    )

    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--res_vol', type=int, default=16)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--z_dim_bg', type=int, default=256)

    parser.add_argument('--eval_inj_idx', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='compcar')

    args = parser.parse_args()
    args.device = device

    if args.dataset == 'compcar':
        # giraffe params
        args.scale_range_min = [0.2, 0.16, 0.16]
        args.scale_range_max = [0.25, 0.2, 0.2]
        args.translation_range_min = [-0.22, -0.12, -0.06]
        args.translation_range_max = [0.22, 0.12, 0.08]
        args.rotation_range = [0., 1.]
        args.range_v = [0.41667, 0.5]
        args.fov = 10
        args.pos_share = True
        args.grf_use_mlp = False
        args.use_viewdirs = True
        args.cid = 7

    elif args.dataset == 'ffhq':
        # giraffe params
        args.scale_range_min = [0.21, 0.21, 0.21]
        args.scale_range_max = [0.21, 0.21, 0.21]
        args.translation_range_min = [0., 0., 0.]
        args.translation_range_max = [0., 0., 0.]
        args.rotation_range = [0.40278, 0.59722]
        args.range_v = [0.4167, 0.5]
        args.fov = 10
        args.pos_share = False
        args.grf_use_mlp = True
        args.use_viewdirs = True
        args.cid = 15

    elif args.dataset == 'cat':
        # giraffe params
        args.scale_range_min = [0.21, 0.21, 0.21]
        args.scale_range_max = [0.21, 0.21, 0.21]
        args.translation_range_min = [0., 0., 0.]
        args.translation_range_max = [0., 0., 0.]
        args.rotation_range = [0.40278, 0.59722]
        args.range_v = [0.4167, 0.5]
        args.fov = 10
        args.pos_share = False
        args.grf_use_mlp = True
        args.use_viewdirs = True
        args.cid = 8

    elif args.dataset == 'celeba':
        # giraffe params
        args.scale_range_min = [0.21, 0.21, 0.21]
        args.scale_range_max = [0.21, 0.21, 0.21]
        args.translation_range_min = [0., 0., 0.]
        args.translation_range_max = [0., 0., 0.]
        args.rotation_range = [0.375, 0.625]
        args.range_v = [0.4167, 0.5]
        args.fov = 10
        args.pos_share = False
        args.grf_use_mlp = True
        args.use_viewdirs = True
        args.cid = 15

    if args.size == 64:
        args.feat_dim = 256
        args.bbox_n_steps = 64
        args.bbox_render_size = 16
    elif args.size == 128:
        args.feat_dim = 256
        args.bbox_n_steps = 64
        args.bbox_render_size = 16
    elif args.size == 256:
        args.feat_dim = 256
        args.bbox_n_steps = 64
        args.bbox_render_size = 16
        args.eval_inj_idx = 2
    elif args.size == 512:
        args.feat_dim = 256
        args.bbox_n_steps = 64
        args.bbox_render_size = 16
        args.eval_inj_idx = 4
    elif args.size == 1024:
        args.feat_dim = 256
        args.bbox_n_steps = 64
        args.bbox_render_size = 16
        args.eval_inj_idx = 4

    generator = GIRAFFEHDGenerator(
        device=device,
        z_dim=args.z_dim,
        z_dim_bg=args.z_dim_bg,
        size=args.size,
        resolution_vol=args.res_vol,
        feat_dim=args.feat_dim,
        range_v=args.range_v,
        fov=args.fov,
        rotation_range=args.rotation_range,
        scale_range_max=args.scale_range_max,
        scale_range_min=args.scale_range_min,
        translation_range_max=args.translation_range_max,
        translation_range_min=args.translation_range_min,
        refine_n_styledconv=2,
        refine_kernal_size=3,
        grf_use_mlp=args.grf_use_mlp,
        pos_share=args.pos_share,
        use_viewdirs=args.use_viewdirs
    ).to(device)

    print('load model:', args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt['g_ema'])

    generator = nn.DataParallel(generator, device_ids=[0])
    generator.requires_grad_(False)
    generator.eval()

    g_module = generator.module

    seg_net = deeplabv3_resnet101(pretrained=True, progress=False).to(device)
    seg_net.requires_grad_(False)
    seg_net.eval()

    change_fg_score = 0

    args.n_sample = args.n_sample // args.batch * args.batch
    batch_li = args.n_sample // args.batch * [args.batch]
    noises = g_module.make_noise(args.batch, device)

    for batch in tqdm(batch_li):
        img_rep = g_module.get_rand_rep(batch)
        img_li = generator(batch, img_rep=img_rep, noises=noises, inject_index=args.eval_inj_idx, mode='eval', return_ids=[])
        img0 = img_li[0]
        bg0 = img_li[2]
        img0 = norm_range(img0)
        mask0 = get_mask(seg_net, img0, args.cid).unsqueeze(1)

        img_rep = latent_change_fg(g_module, img_rep)
        img1 = generator(batch, img_rep=img_rep, noises=noises, inject_index=args.eval_inj_idx, mode='eval', _bg_img=bg0)[0]
        img1 = norm_range(img1)
        mask1 = get_mask(seg_net, img1, args.cid).unsqueeze(1)

        mutual_bg_mask = (1-mask0) * (1-mask1)

        diff = F.l1_loss(mutual_bg_mask*img1, mutual_bg_mask*img0, reduction='none')
        diff = torch.where(diff < 1/255, torch.zeros_like(diff), torch.ones_like(diff))
        diff = torch.sum(diff, dim=1)
        diff = torch.where(diff < 1, torch.zeros_like(diff), torch.ones_like(diff))

        utils.save_image(
            (1-mask1)*img1,
            f'temp/mask1.png',
            nrow=8,
            normalize=True,
            range=(0, 1),
            padding=0,
        )
        utils.save_image(
            (1-mask0)*img0,
            f'temp/mask0.png',
            nrow=8,
            normalize=True,
            range=(0, 1),
            padding=0,
        )
        utils.save_image(
            img0,
            f'temp/img0.png',
            nrow=8,
            normalize=True,
            range=(0, 1),
            padding=0,
        )
        utils.save_image(
            img1,
            f'temp/img1.png',
            nrow=8,
            normalize=True,
            range=(0, 1),
            padding=0,
        )
        utils.save_image(
            diff.unsqueeze(1),
            f'temp/diff.png',
            nrow=8,
            normalize=True,
            range=(0, 1),
            padding=0,
        )
        sys.exit()

        change_fg_score += torch.sum(torch.sum(diff, dim=(1,2)) / (torch.sum(mutual_bg_mask, dim=(1,2,3))+1e-8))

    print(f'change_fg_score: {change_fg_score/args.n_sample}')
