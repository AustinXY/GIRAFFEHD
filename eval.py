import os
import sys
import argparse
import torch
from torchvision import utils
from model import GIRAFFEHDGenerator

def get_interval(args):
    if args.control_i == 4:
        if args.range_u is None:
            p0 = torch.tensor([[args.ckpt_args.range_u[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.ckpt_args.range_u[1]]]).repeat(
                args.batch, 1).to(args.device)
        else:
            p0 = torch.tensor([[args.range_u[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.range_u[1]]]).repeat(
                args.batch, 1).to(args.device)

    elif args.control_i == 5:
        if args.range_v is None:
            p0 = torch.tensor([[args.ckpt_args.range_v[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.ckpt_args.range_v[1]]]).repeat(
                args.batch, 1).to(args.device)
        else:
            p0 = torch.tensor([[args.range_v[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.range_v[1]]]).repeat(
                args.batch, 1).to(args.device)

    elif args.control_i == 6:
        print('Changing radius not implemented')
        sys.exit()

    elif args.control_i == 7:
        if args.scale_range_min is None:
            p0 = torch.tensor([args.ckpt_args.scale_range_min]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([args.ckpt_args.scale_range_max]).repeat(
                args.batch, 1).to(args.device)
        else:
            p0 = torch.tensor([args.scale_range_min]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([args.scale_range_max]).repeat(
                args.batch, 1).to(args.device)

    elif args.control_i == 8:
        if args.translation_range_min is None:
            p0 = torch.tensor([args.ckpt_args.translation_range_min]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([args.ckpt_args.translation_range_max]).repeat(
                args.batch, 1).to(args.device)
        else:
            p0 = torch.tensor([args.translation_range_min]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([args.translation_range_max]).repeat(
                args.batch, 1).to(args.device)

    elif args.control_i == 9:
        if args.rotation_range is None:
            p0 = torch.tensor([[args.ckpt_args.rotation_range[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.ckpt_args.rotation_range[1]]]).repeat(
                args.batch, 1).to(args.device)
        else:
            p0 = torch.tensor([[args.rotation_range[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.rotation_range[1]]]).repeat(
                args.batch, 1).to(args.device)

    elif args.control_i == 10:
        if args.bg_translation_range_min is None:
            p0 = torch.tensor([args.ckpt_args.bg_translation_range_min]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([args.ckpt_args.bg_translation_range_max]).repeat(
                args.batch, 1).to(args.device)
        else:
            p0 = torch.tensor([args.bg_translation_range_min]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([args.bg_translation_range_max]).repeat(
                args.batch, 1).to(args.device)

    elif args.control_i == 11:
        if args.bg_rotation_range is None:
            p0 = torch.tensor([[args.ckpt_args.bg_rotation_range[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.ckpt_args.bg_rotation_range[1]]]).repeat(
                args.batch, 1).to(args.device)
        else:
            p0 = torch.tensor([[args.bg_rotation_range[0]]]).repeat(
                args.batch, 1).to(args.device)
            p1 = torch.tensor([[args.bg_rotation_range[1]]]).repeat(
                args.batch, 1).to(args.device)

    return p0, p1


def eval(args, generator):
    generator.eval()

    img_li = []
    if args.control_i in list(range(0,4)):
        img_rep = generator.get_rand_rep(args.batch)
        for i in range(args.n_sample):
            _img_rep = generator.get_rand_rep(args.batch)
            img_rep[args.control_i] = _img_rep[args.control_i]
            img = generator(img_rep=img_rep, inject_index=args.inj_idx, mode='eval')[0]
            img_li.append(img)

    if args.control_i in list(range(4,12)):
        p0, p1 = get_interval(args)
        delta = (p1 - p0) / (args.n_sample - 1)

        img_rep = generator.get_rand_rep(args.batch)
        for i in range(args.n_sample):
            p = p0 + delta * i
            img_rep[args.control_i] = p
            img = generator(img_rep=img_rep, inject_index=args.inj_idx, mode='eval')[0]
            img_li.append(img)

    img = []
    for b in range(args.batch):
        b_li = []
        for i in range(args.n_sample):
            b_li.append(img_li[i][b:b+1])

        img.append(torch.cat(b_li))
    img = torch.cat(img)

    utils.save_image(
        img,
        f"eval/control_{args.control_i}.png",
        nrow=args.n_sample,
        normalize=True,
        range=(-1, 1),
    )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Giraffe trainer")

    parser.add_argument('--ckpt', type=str, default=None, help='path to the checkpoint')

    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--n_sample', type=int, default=8, help='number of the samples generated')
    parser.add_argument('--inj_idx', type=int, default=-1, help='inject index for evaluation')

    parser.add_argument("--control_i", type=int, default=0, help='control index')
    # 0: fg_shape; 1: fg_app; 2: bg_shape;    3: bg_app;   4: camera rotation angle;  5: elevation angle;
    # --: radius;  7: scale;  8: translation; 9: rotation; 10: bg translation;        11: bg rotation;

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.device = device

    assert args.ckpt is not None
    print("load model:", args.ckpt)

    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    args.ckpt_args = ckpt['args']

    # change interpolation ranges if needed
    args.scale_range_min = None  # [0.2, 0.16, 0.16]
    args.scale_range_max = None  # [0.25, 0.2, 0.2]

    args.translation_range_min = None  # [-0.22, -0.12, -0.06]
    args.translation_range_max = None  # [0.22, 0.12, 0.08]

    args.rotation_range = None  # [0., 1.]

    args.bg_translation_range_min = None  # [-0.2, -0.2, 0.]
    args.bg_translation_range_max = None  # [0.2, 0.2, 0.]

    args.bg_rotation_range = None  # [0., 0.]

    args.range_u = None  # [0., 0.]

    args.range_v = None  # [0.41667, 0.5]


    if args.inj_idx == -1:
        if args.ckpt_args.size == 256:
            args.inj_idx = 2
        elif args.ckpt_args.size == 512:
            args.inj_idx = 4
        elif args.ckpt_args.size == 1024:
            args.inj_idx = 4

    generator = GIRAFFEHDGenerator(
        device=device,
        z_dim=args.ckpt_args.z_dim,
        z_dim_bg=args.ckpt_args.z_dim_bg,
        size=args.ckpt_args.size,
        resolution_vol=args.ckpt_args.res_vol,
        feat_dim=args.ckpt_args.feat_dim,
        range_u=args.ckpt_args.range_u,
        range_v=args.ckpt_args.range_v,
        fov=args.ckpt_args.fov,
        scale_range_max=args.ckpt_args.scale_range_max,
        scale_range_min=args.ckpt_args.scale_range_min,
        translation_range_max=args.ckpt_args.translation_range_max,
        translation_range_min=args.ckpt_args.translation_range_min,
        rotation_range=args.ckpt_args.rotation_range,
        bg_translation_range_max=args.ckpt_args.bg_translation_range_max,
        bg_translation_range_min=args.ckpt_args.bg_translation_range_min,
        bg_rotation_range=args.ckpt_args.bg_rotation_range,
        refine_n_styledconv=2,
        refine_kernal_size=3,
        grf_use_mlp=args.ckpt_args.grf_use_mlp,
        pos_share=args.ckpt_args.pos_share,
        use_viewdirs=args.ckpt_args.use_viewdirs,
        grf_use_z_app=args.ckpt_args.grf_use_z_app,
        fg_gen_mask=args.ckpt_args.fg_gen_mask
    ).to(device)

    generator.load_state_dict(ckpt["g_ema"])

    with torch.no_grad():
        eval(args, generator)
