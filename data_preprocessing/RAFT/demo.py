import sys

sys.path.append("core")

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = "cuda"


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, img_name=None):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imwrite(f"{img_name}", img_flo[:, :, [2, 1, 0]])


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.outdir_conf, exist_ok=True)
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, "*.png")) + glob.glob(
            os.path.join(args.path, "*.jpg")
        )

        images = sorted(images)
        i = 0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            if args.if_mask:
                mk_file1 = imfile1.split("/")
                mk_file1[-2] = f"{args.name}_masks"
                mk_file1 = "/".join(mk_file1)
                mk_file2 = imfile2.split("/")
                mk_file2[-2] = f"{args.name}_masks"
                mk_file2 = "/".join(mk_file2)
                mask1 = cv2.imread(mk_file1.replace("jpg", "png"), 0)
                mask2 = cv2.imread(mk_file2.replace("jpg", "png"), 0)
                mask1 = torch.from_numpy(mask1).to(DEVICE).float()
                mask2 = torch.from_numpy(mask2).to(DEVICE).float()
                mask1[mask1 > 0] = 1
                mask2[mask2 > 0] = 1
                image1 *= mask1
                image2 *= mask2

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            if args.if_mask:
                mask1, mask2 = padder.pad(
                    mask1.unsqueeze(0).unsqueeze(0), mask2.unsqueeze(0).unsqueeze(0)
                )
                mask1 = mask1.squeeze()
                mask2 = mask2.squeeze()

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_low_, flow_up_ = model(image2, image1, iters=20, test_mode=True)
            flow_1to2 = flow_up.clone()
            flow_2to1 = flow_up_.clone()

            _, _, H, W = image1.shape
            x = torch.linspace(0, 1, W)
            y = torch.linspace(0, 1, H)
            grid_x, grid_y = torch.meshgrid(x, y)
            grid = torch.stack([grid_x, grid_y], dim=0).to(DEVICE)
            grid = grid.permute(0, 2, 1)
            grid[0] *= W
            grid[1] *= H
            if args.if_mask:
                flow_up[:, :, mask1.long() == 0] = 10000
            grid_ = grid + flow_up.squeeze()

            grid_norm = grid_.clone()
            grid_norm[0, ...] = 2 * grid_norm[0, ...] / (W - 1) - 1
            grid_norm[1, ...] = 2 * grid_norm[1, ...] / (H - 1) - 1

            flow_bilinear_ = F.grid_sample(
                flow_up_,
                grid_norm.unsqueeze(0).permute(0, 2, 3, 1),
                mode="bilinear",
                padding_mode="zeros",
            )

            rgb_bilinear_ = F.grid_sample(
                image2,
                grid_norm.unsqueeze(0).permute(0, 2, 3, 1),
                mode="bilinear",
                padding_mode="zeros",
            )
            rgb_np = rgb_bilinear_.squeeze().permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            cv2.imwrite(f"{args.outdir}/warped.png", rgb_np)

            if args.confidence:
                ### Calculate confidence map using cycle consistency.
                # 1). First calculate `warped_image2` by the following formula:
                #   warped_image2 = F.grid_sample(image1, flow_2to1)
                # 2). Then calculate `warped_image1` by the following formula:
                #   warped_image1 = F.grid_sample(warped_image2, flow_1to2)
                # 3) Finally calculate the confidence map:
                #  confidence_map = metric_func(image1 - warped_image1)

                grid_2to1 = grid + flow_2to1.squeeze()
                norm_grid_2to1 = grid_2to1.clone()
                norm_grid_2to1[0, ...] = 2 * norm_grid_2to1[0, ...] / (W - 1) - 1
                norm_grid_2to1[1, ...] = 2 * norm_grid_2to1[1, ...] / (H - 1) - 1
                warped_image2 = F.grid_sample(
                    image1,
                    norm_grid_2to1.unsqueeze(0).permute(0, 2, 3, 1),
                    mode="bilinear",
                    padding_mode="zeros",
                )

                grid_1to2 = grid + flow_1to2.squeeze()
                norm_grid_1to2 = grid_1to2.clone()
                norm_grid_1to2[0, ...] = 2 * norm_grid_1to2[0, ...] / (W - 1) - 1
                norm_grid_1to2[1, ...] = 2 * norm_grid_1to2[1, ...] / (H - 1) - 1
                warped_image1 = F.grid_sample(
                    warped_image2,
                    norm_grid_1to2.unsqueeze(0).permute(0, 2, 3, 1),
                    mode="bilinear",
                    padding_mode="zeros",
                )

                error = torch.abs(image1 - warped_image1)
                confidence_map = torch.mean(error, dim=1, keepdim=True)
                confidence_map[confidence_map < args.thres] = 1
                confidence_map[confidence_map >= args.thres] = 0

            grid_bck = grid + flow_up.squeeze() + flow_bilinear_.squeeze()
            res = grid - grid_bck
            res = torch.norm(res, dim=0)
            mk = (res < 10) & (flow_up.norm(dim=1).squeeze() > 5)

            pts_src = grid[:, mk]

            pts_dst = grid[:, mk] + flow_up.squeeze()[:, mk]

            pts_src = pts_src.permute(1, 0).cpu().numpy()
            pts_dst = pts_dst.permute(1, 0).cpu().numpy()
            indx = torch.randperm(pts_src.shape[0])[:30]
            # use cv2 to draw the matches in image1 and image2
            img_new = np.zeros((H, W * 2, 3), dtype=np.uint8)
            img_new[:, :W, :] = image1[0].permute(1, 2, 0).cpu().numpy()
            img_new[:, W:, :] = image2[0].permute(1, 2, 0).cpu().numpy()

            for j in indx:
                cv2.line(
                    img_new,
                    (int(pts_src[j, 0]), int(pts_src[j, 1])),
                    (int(pts_dst[j, 0]) + W, int(pts_dst[j, 1])),
                    (0, 255, 0),
                    1,
                )

            cv2.imwrite(f"{args.outdir}/matches.png", img_new)

            np.save(f"{args.outdir}/{i:06d}.npy", flow_up.cpu().numpy())
            if args.confidence:
                np.save(
                    f"{args.outdir_conf}/{i:06d}_c.npy", confidence_map.cpu().numpy()
                )
            i += 1

            viz(image1, flow_up, f"{args.outdir}/flow_up{i:03d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--path", help="dataset for evaluation")
    parser.add_argument("--outdir", help="directory for the ouput the result")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--if_mask",
        action="store_true",
        help="if using the image mask to mask the color img",
    )
    parser.add_argument(
        "--confidence", action="store_true", help="if saving the confidence map"
    )
    parser.add_argument(
        "--discrete",
        action="store_true",
        help="if saving the confidence map in discrete",
    )
    parser.add_argument("--thres", default=4, help="Threshold value for confidence map")
    parser.add_argument("--outdir_conf", help="directory to save flow confidence")
    parser.add_argument("--name", help="the name of a sequence")
    args = parser.parse_args()

    demo(args)
