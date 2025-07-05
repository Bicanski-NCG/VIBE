# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm
import numpy as np
import gc

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize, NormalizeImage, PrepareForNet

from utils.util import compute_scale_and_shift, get_interpolate_frames

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)

    def forward(self, x):
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.head(features, patch_h, patch_w, T)
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)) # return shape [B, T, H, W]
    
    def infer_video_depth(self, frames, target_fps, input_size=518,
                      device='cuda', fp32=False):
        """
        frames:  5-D ndarray  (B, T, H, W, 3)    OR 4-D (T, H, W, 3) for a single clip
        """

        # ---------------- handle batch or single clip ----------------
        if frames.ndim == 4:                     # (T, H, W, 3)
            frames = frames[None]                # add batch dim → (1, T, H, W, 3)
        B, T, H0, W0, _ = frames.shape
        assert T == INFER_LEN, f"expected 32 frames, got {T}"
        frame_height, frame_width = H0, W0

        # ---------- resize policy & Compose (same as before) ----------
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose([
            Resize(
                width=input_size, height=input_size, keep_aspect_ratio=True,
                ensure_multiple_of=14, resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        # --------------- vectorised preprocessing --------------------
        clips = []
        for b in range(B):
            clip = torch.stack(
                [torch.from_numpy(
                    transform({'image': frames[b, t].astype(np.float32) / 255.0})['image']
                ) for t in range(T)],
                dim=0                               # time axis
            )                                       # (T, C, H', W')
            clips.append(clip)
        x = torch.stack(clips, dim=0).to(device)    # (B, T, C, H', W')

        # ----------------------- forward pass ------------------------
        with torch.autocast(device_type=device, enabled=(not fp32)):
            depth = self.forward(x)                 # (B, T, Hd, Wd)

        return  