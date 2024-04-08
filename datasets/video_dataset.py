import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from einops import rearrange, reduce, repeat
from torchvision import transforms as T
import glob
import cv2


# The basic dataset of reading rays
class VideoDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        img_wh=(504, 378),
        mask_dir=None,
        flow_dir=None,
        canonical_wh=None,
        ref_idx=None,
        canonical_dir=None,
        test=False,
    ):
        self.test = test
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.mask_dir = mask_dir
        self.flow_dir = flow_dir
        self.canonical_wh = canonical_wh
        self.ref_idx = ref_idx
        self.canonical_dir = canonical_dir
        self.read_meta()

    def read_meta(self):
        all_images_path = []
        self.ts_w = []
        self.all_images = []
        h = self.img_wh[1]
        w = self.img_wh[0]
        # construct grid
        grid = np.indices((h, w)).astype(np.float32)
        # normalize
        grid[0, :, :] = grid[0, :, :] / h
        grid[1, :, :] = grid[1, :, :] / w
        self.grid = torch.from_numpy(rearrange(grid, "c h w -> (h w) c"))
        warp_code = 1
        for input_image_path in sorted(glob.glob(f"{self.root_dir}/*")):
            print(input_image_path)
            all_images_path.append(input_image_path)
            self.ts_w.append(torch.Tensor([warp_code]).long())
            warp_code += 1

        if self.canonical_wh:
            h_c = self.canonical_wh[1]
            w_c = self.canonical_wh[0]
            grid_c = np.indices((h_c, w_c)).astype(np.float32)
            grid_c[0, :, :] = (grid_c[0, :, :] - (h_c - h) / 2) / h
            grid_c[1, :, :] = (grid_c[1, :, :] - (w_c - w) / 2) / w
            self.grid_c = torch.from_numpy(rearrange(grid_c, "c h w -> (h w) c"))
        else:
            self.grid_c = self.grid
            self.canonical_wh = self.img_wh

        if self.mask_dir:
            self.all_masks = []
        if self.flow_dir:
            self.all_flows = []
        else:
            self.all_flows = None

        if self.split == "train" or self.split == "val":
            if self.canonical_dir is not None:
                all_images_path_ = sorted(glob.glob(f"{self.canonical_dir}/*.png"))
                self.canonical_img = []
                for input_image_path in all_images_path_:
                    input_image = cv2.imread(input_image_path)
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                    input_image = cv2.resize(
                        input_image,
                        (self.canonical_wh[0], self.canonical_wh[1]),
                        interpolation=cv2.INTER_AREA,
                    )
                    input_image_tensor = torch.from_numpy(input_image).float() / 256
                    self.canonical_img.append(input_image_tensor)
                self.canonical_img = torch.stack(self.canonical_img, dim=0)

            for input_image_path in all_images_path:
                input_image = cv2.imread(input_image_path)
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_image = cv2.resize(
                    input_image,
                    (self.img_wh[0], self.img_wh[1]),
                    interpolation=cv2.INTER_AREA,
                )
                input_image_tensor = torch.from_numpy(input_image).float() / 256
                self.all_images.append(input_image_tensor)
                if self.mask_dir:
                    input_image_name = input_image_path.split("/")[-1][:-4]
                    for i in range(len(self.mask_dir)):
                        input_mask = cv2.imread(
                            f"{self.mask_dir[i]}/{input_image_name}.png"
                        )
                        input_mask = cv2.resize(
                            input_mask,
                            (self.img_wh[0], self.img_wh[1]),
                            interpolation=cv2.INTER_AREA,
                        )
                        input_mask_tensor = torch.from_numpy(input_mask).float() / 256
                        self.all_masks.append(input_mask_tensor)

        if self.split == "val":
            input_image = cv2.imread(all_images_path[0])
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = cv2.resize(
                input_image,
                (self.img_wh[0], self.img_wh[1]),
                interpolation=cv2.INTER_AREA,
            )
            input_image_tensor = torch.from_numpy(input_image).float() / 256
            self.all_images.append(input_image_tensor)
            if self.mask_dir:
                input_image_name = all_images_path[0].split("/")[-1][:-4]
                for i in range(len(self.mask_dir)):
                    input_mask = cv2.imread(
                        f"{self.mask_dir[i]}/{input_image_name}.png"
                    )
                    input_mask = cv2.resize(
                        input_mask,
                        (self.img_wh[0], self.img_wh[1]),
                        interpolation=cv2.INTER_AREA,
                    )
                    input_mask_tensor = torch.from_numpy(input_mask).float() / 256
                    self.all_masks.append(input_mask_tensor)

        if self.flow_dir:
            for input_image_path in sorted(glob.glob(f"{self.flow_dir}/*npy")):
                flow_load = np.load(input_image_path)  # (1, 2, h, w)
                flow_tensor = torch.from_numpy(flow_load).float()[:, [1, 0]]
                flow_tensor = torch.nn.functional.interpolate(
                    flow_tensor, size=(self.img_wh[1], self.img_wh[0])
                )
                H_, W_ = flow_load.shape[-2], flow_load.shape[-1]
                flow_tensor = flow_tensor.reshape(2, -1).transpose(1, 0)
                flow_tensor[..., 0] /= W_
                flow_tensor[..., 1] /= H_
                self.all_flows.append(flow_tensor)

            i = 0
            for input_image_path in sorted(
                glob.glob(f"{self.flow_dir}_confidence/*npy")
            ):
                flow_load = np.load(input_image_path)
                flow_tensor = torch.from_numpy(flow_load).float()
                flow_tensor = torch.nn.functional.interpolate(
                    flow_tensor, size=(self.img_wh[1], self.img_wh[0])
                )
                flow_tensor = flow_tensor.reshape(1, -1).transpose(1, 0)
                flow_tensor = flow_tensor.sum(dim=-1) < 0.05
                self.all_flows[i][flow_tensor] = 5
                i += 1

        if self.split == "val":
            self.ref_idx = 0

    def __len__(self):
        if self.test:
            return len(self.all_images)
        return 200 * len(self.all_images)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "val":
            idx = idx % len(self.all_images)
            sample = {
                "rgbs": self.all_images[idx],
                "canonical_img": (
                    self.all_images[idx]
                    if self.canonical_dir is None
                    else self.canonical_img
                ),
                "ts_w": self.ts_w[idx],
                "grid": self.grid,
                "canonical_wh": self.canonical_wh,
                "img_wh": self.img_wh,
                "masks": (
                    self.all_masks[
                        len(self.mask_dir) * idx : len(self.mask_dir) * idx
                        + len(self.mask_dir)
                    ]
                    if self.mask_dir
                    else [torch.ones((self.img_wh[1], self.img_wh[0], 1))]
                ),
                "flows": (
                    self.all_flows[idx]
                    if (idx < len(self.all_images) - 2) & (self.all_flows is not None)
                    else -1e5
                ),
                "grid_c": self.grid_c,
                "reference": (
                    [
                        self.all_images[self.ref_idx],
                        self.all_masks[
                            len(self.mask_dir) * idx : len(self.mask_dir) * idx
                            + len(self.mask_dir)
                        ],
                    ]
                    if not self.ref_idx is None
                    else -1e5
                ),
                "seq_len": len(self.all_images),
            }

        return sample
