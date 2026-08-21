import os
import math
import numpy as np
import imageio.v2 as imageio
from PIL import Image

import torch
import torchvision
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only


def save_video_grid(video, filename, nrow=None, fps=6):
    batch, channels, frames, height, width = video.shape
    video = video.permute(0, 2, 3, 4, 1).contiguous()
    video = (video.detach().cpu().numpy() * 255).astype("uint8")
    if nrow is None:
        nrow = math.ceil(math.sqrt(batch))
    ncol = math.ceil(batch / nrow)
    padding = 1
    grid = np.zeros(
        (
            frames,
            (padding + height) * nrow + padding,
            (padding + width) * ncol + padding,
            channels,
        ),
        dtype="uint8",
    )
    for index in range(batch):
        row = index // ncol
        column = index % ncol
        start_row = (padding + height) * row
        start_column = (padding + width) * column
        grid[
            :,
            start_row : start_row + height,
            start_column : start_column + width,
        ] = video[index]
    imageio.mimsave(filename, list(grid), fps=fps)

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp


    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        # print(root)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = grid + 0.5
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, trainer, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -0.5, 0.5)

            self.log_local(trainer.default_root_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(trainer, pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.log_img(trainer, pl_module, batch, batch_idx, split="val")




class VideoLogger(Callback):
    def __init__(self, batch_frequency, max_videos, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_videos = max_videos
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp


    @rank_zero_only
    def log_local(self, save_dir, split, videos,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "videos", split)
        for k in videos:
            grid = videos[k] + 0.5
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.mp4".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            save_video_grid(grid, path)

    def log_vid(self, trainer, pl_module, batch, batch_idx, split="train"):
        # print(batch_idx, self.batch_freq, self.check_frequency(batch_idx) and hasattr(pl_module, "log_videos") and callable(pl_module.log_videos) and self.max_videos > 0)
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_videos") and
                callable(pl_module.log_videos) and
                self.max_videos > 0):
            # print(batch_idx, self.batch_freq,  self.check_frequency(batch_idx))
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                videos = pl_module.log_videos(batch, split=split, batch_idx=batch_idx)

            for k in videos:
                N = min(videos[k].shape[0], self.max_videos)
                videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().cpu()
                    if self.clamp:
                        videos[k] = torch.clamp(videos[k], -0.5, 0.5)

            self.log_local(trainer.default_root_dir, split, videos,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_vid(trainer, pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.log_vid(trainer, pl_module, batch, batch_idx, split="val")
