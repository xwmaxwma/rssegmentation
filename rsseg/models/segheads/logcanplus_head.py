# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange
from rsseg.models.basemodules.dysample import DySample

def patch_split(input, patch_size):
    """
    input: (B, C, H, W)
    output: (B*num_h*num_w, C, patch_h, patch_w)
    """
    B, C, H, W = input.size()
    num_h, num_w = patch_size
    patch_h, patch_w = H // num_h, W // num_w
    out = input.view(B, C, num_h, patch_h, num_w, patch_w)
    out = out.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, patch_h,
                                                          patch_w)  # (B*num_h*num_w, C, patch_h, patch_w)
    return out


def patch_recover(input, patch_size):
    """
    input: (B*num_h*num_w, C, patch_h, patch_w)
    output: (B, C, H, W)
    """
    N, C, patch_h, patch_w = input.size()
    num_h, num_w = patch_size
    H, W = num_h * patch_h, num_w * patch_w
    B = N // (num_h * num_w)

    out = input.view(B, num_h, num_w, C, patch_h, patch_w)
    out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return out




class RVSA_MRAM(nn.Module):
    def __init__(self, dim, num_heads,
                 num_classes,out_dim=None, qkv_bias=True, qk_scale=None,
                learnable=True,
                restart_regression=True,num_deform=None,
                patch_size = (4,4)):
        super().__init__()

        self.feat_decoder = nn.Conv2d(dim, num_classes, kernel_size=1)
        self.patch_size = patch_size

        self.num_heads = num_heads
        self.dim = dim
        out_dim = out_dim or dim
        self.out_dim = out_dim
        head_dim = dim // self.num_heads

        self.learnable = learnable
        self.restart_regression = restart_regression
        if self.learnable:
            # if num_deform is None, we set num_deform to num_heads as default

            if num_deform is None:
                num_deform = 1
            self.num_deform = num_deform

            self.sampling_offsets = nn.Sequential(
                nn.AdaptiveAvgPool2d([1,1]),
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 2, kernel_size=1, stride=1)
            )
            self.sampling_scales = nn.Sequential(
                nn.AdaptiveAvgPool2d([1,1]),
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 2, kernel_size=1, stride=1)
            )
            # add angle
            self.sampling_angles = nn.Sequential(
                nn.AdaptiveAvgPool2d([1,1]),
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 1, kernel_size=1, stride=1)
            )
            self._reset_parameters()

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.out_project = nn.Linear(dim, out_dim, bias=qkv_bias)

        self.cat_conv = nn.Sequential(
            conv_3x3(out_dim * 2, out_dim),
            nn.Dropout2d(0.1),
            conv_3x3(out_dim, out_dim),
            nn.Dropout2d(0.1)
        )

    '''
    x:[b,c,h,w]
    global_center:[b,k,c]
    '''
    def forward(self, x,global_center):
        shortcut = x
        B, C, H, W = x.shape
        probs = self.feat_decoder(x)
        K = probs.shape[1]
        probs = probs.view(B, K, -1)  # batch * k * hw
        probs = F.softmax(probs, dim=2).reshape(B,K,H,W)


        patch_x = patch_split(x,self.patch_size)

        window_size_h = patch_x.shape[-2]
        window_size_w = patch_x.shape[-1]

        # padding on left-right-up-down
        expand_h, expand_w = H, W

        # window num in padding features
        window_num_h = self.patch_size[0]
        window_num_w = self.patch_size[1]

        #前面这些操作是为了补齐窗口大小
        image_reference_h = torch.linspace(-1, 1, expand_h).to(x.device)
        image_reference_w = torch.linspace(-1, 1, expand_w).to(x.device)
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0)\
            .permute(0, 2,1).unsqueeze(0)  # 1, 2, H, W

        # position of the window relative to the image center
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=[window_size_h,window_size_w])  # 1,2, nh, nw
        image_reference = image_reference.reshape(1, 2, window_num_h, window_size_h, window_num_w,
                                                  window_size_w)  # 1, 2, nh, ws, nw, ws
        assert window_num_h == window_reference.shape[-2]
        assert window_num_w == window_reference.shape[-1]

        window_reference = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)  # 1,2, nh,1, nw,1
        # coords of pixels in each window

        base_coords_h = torch.arange(window_size_h).to(x.device) * 2 * window_size_h / window_size_h / (expand_h - 1)  # ws
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(window_size_w).to(x.device) * 2 * window_size_w / window_size_w / (expand_w - 1)
        base_coords_w = (base_coords_w - base_coords_w.mean())
        # base_coords = torch.stack(torch.meshgrid(base_coords_w, base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, 1, self.attn_ws, 1, self.attn_ws)

        # extend to each window
        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)  # ws -> 1,ws -> nh,ws
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == window_size_h

        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)  # nw,ws
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == window_size_w
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)  # nh*ws
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)  # nw*ws

        window_coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2,
                                                                                                               1).reshape(
            1, 2, window_num_h, window_size_h, window_num_w, window_size_w)  # 1, 2, nh, ws, nw, ws
        # base_coords = window_reference+window_coords
        base_coords = image_reference

        if self.restart_regression:
            # compute for each head in each batch
            coords = base_coords.repeat(B * self.num_heads, 1, 1, 1, 1, 1)  # B*nH, 2, nh, ws, nw, ws
        if self.learnable:
            num_predict_total = B * self.num_heads * self.num_deform

            # offset factors
            sampling_offsets = self.sampling_offsets(patch_x) #[b*ph*pw,num_heads * 2,1]
            sampling_offsets = sampling_offsets.reshape(B,window_num_h , window_num_w,self.num_heads * self.num_deform,2)
            sampling_offsets = sampling_offsets.permute(0,3,4,1,2).contiguous()
            sampling_offsets = sampling_offsets.reshape(num_predict_total,2, window_num_h , window_num_w)

            #归一化
            sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (H // window_num_h)
            sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (W // window_num_w)

            # scale fators
            sampling_scales = self.sampling_scales(patch_x)  # B, heads*2, h // window_size, w // window_size
            sampling_scales = sampling_scales.reshape(B,window_num_h , window_num_w,self.num_heads * self.num_deform,2)
            sampling_scales = sampling_scales.permute(0,3,4,1,2).contiguous()
            sampling_scales = sampling_scales.reshape(num_predict_total,2, window_num_h , window_num_w)

            # rotate factor
            sampling_angle = self.sampling_angles(patch_x)
            sampling_angle = sampling_angle.reshape(B,window_num_h , window_num_w,self.num_heads * self.num_deform,1)
            sampling_angle = sampling_angle.permute(0,3,4,1,2).contiguous()
            sampling_angle = sampling_angle.reshape(num_predict_total,1,window_num_h , window_num_w)

            # first scale
            window_coords = window_coords * (sampling_scales[:, :, :, None, :, None] + 1)

            # then rotate around window center

            window_coords_r = window_coords.clone()

            # 0:x,column, 1:y,row

            window_coords_r[:, 0, :, :, :, :] = -window_coords[:, 1, :, :, :, :] * torch.sin(
                sampling_angle[:, 0, :, None, :, None]) + window_coords[:, 0, :, :, :, :] * torch.cos(
                sampling_angle[:, 0, :, None, :, None])
            window_coords_r[:, 1, :, :, :, :] = window_coords[:, 1, :, :, :, :] * torch.cos(
                sampling_angle[:, 0, :, None, :, None]) + window_coords[:, 0, :, :, :, :] * torch.sin(
                sampling_angle[:, 0, :, None, :, None])

            # system transformation: window center -> image center

            coords = window_reference + window_coords_r + sampling_offsets[:, :, :, None, :, None]

        # final offset
        sample_coords = coords.permute(0, 2, 3, 4, 5, 1).contiguous().reshape(num_predict_total, window_size_h * window_num_h,
                                                                 window_size_w * window_num_w, 2)
        
        #[b,h,w,c]
        logcal_key = self.k(x.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()

        transform_x = F.grid_sample(
            logcal_key.reshape(num_predict_total,
                      self.dim // self.num_heads // self.num_deform,
                      H,W),
            grid=sample_coords, padding_mode='zeros', align_corners=True
        ).reshape(num_predict_total,-1,
                  window_num_h, window_size_h,
                  window_num_w, window_size_w).permute(0,2,4,3,5,1).contiguous().reshape(-1,window_size_w*window_size_h,self.dim // self.num_heads // self.num_deform) #[B*wnh,wh*ww,C]
        transform_probs = F.grid_sample(
            probs.repeat(num_predict_total // B,1,1,1),
            grid=sample_coords, padding_mode='zeros', align_corners=True
        ).reshape(num_predict_total,K,
                  window_num_h, window_size_h,
                  window_num_w, window_size_w).permute(0,2,4,3,5,1).contiguous().reshape(-1,window_size_w*window_size_h,K) #[B*wnh,wh*ww,K]


        #计算局部类中心作为key
        logcal_center = torch.matmul(transform_probs.permute(0,2,1),transform_x) #[B,K,C]
        logcal_center = logcal_center.reshape(B * window_num_w * window_num_h,self.num_heads,K,-1)
        query = x.reshape(B,C,
                  window_num_h, window_size_h,
                  window_num_w, window_size_w).permute(0,2,4,3,5,1).contiguous().reshape(-1,window_size_w*window_size_h,C) #[B*wnh,wh*ww,C]

        #[b,num_heads,c,hw]


        query = self.q(query).permute(0,2,1).contiguous().reshape(
            B * window_num_w * window_num_h,
            self.num_heads,C // self.num_heads,
            window_size_w*window_size_h) #[B,nds,c,wh*ww]

        value = global_center.repeat(window_num_w * window_num_h,1,1)
        value = self.v(value).permute(0,2,1).contiguous().reshape(
            B * window_num_w * window_num_h,self.num_heads,
            C // self.num_heads,K)#[B,nds,c,k]


        query = query.permute(0,1,3,2).contiguous() #[B,nds,wh*ww,c]
        dots = (query @ logcal_center.permute(0,1,3,2).contiguous()) * self.scale #[B,nds,wh*ww,K]


        attn = dots.softmax(dim=-1)
        out = attn @ value.permute(0,1,3,2).contiguous() #[B,num_heads,HW,C]

        context = out.permute(0, 1, 3, 2).contiguous() #[B,num_heads,c,wh*ww]
        context = context.reshape(B * window_num_w * window_num_h, C,-1).permute(0,2,1).contiguous()  # (B,wh*ww,C)

        context = self.out_project(context).permute(0,2,1).contiguous().reshape(-1,C,window_size_h,window_size_w)  # (B*num_h*num_w, C, wh, ww)

        context = patch_recover(context,self.patch_size)
        
        out = self.cat_conv(torch.cat([shortcut, context], dim=1))
        return out

    def _reset_parameters(self):
        if self.learnable:
            nn.init.constant_(self.sampling_offsets[-1].weight, 0.)
            nn.init.constant_(self.sampling_offsets[-1].bias, 0.)
            nn.init.constant_(self.sampling_scales[-1].weight, 0.)
            nn.init.constant_(self.sampling_scales[-1].bias, 0.)




def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

class SpatialGatherModule(nn.Module):
    def __init__(self, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, features, probs):  # (B*num_h*num_w, C, patch_h, patch_w) (B*num_h*num_w, K, patch_h, patch_w)
        batch_size, num_classes, h, w = probs.size()
        probs = probs.view(batch_size, num_classes, -1)  # batch * k * hw
        probs = F.softmax(self.scale * probs, dim=2)

        features = features.view(batch_size, features.size(1), -1)
        features = features.permute(0, 2, 1)  # batch * hw * c

        ocr_context = torch.matmul(probs, features)  # (B, k, c)
        return ocr_context


def upsample_add(x_small, x_big):
    x_small = F.interpolate(x_small, scale_factor=2, mode="bilinear", align_corners=False)
    return torch.cat([x_small, x_big], dim=1)


class LoGCANPlus_Head(nn.Module):

    def __init__(self,
                 transform_channel,
                 in_channel,
                 num_class,
                 num_heads,
                 patch_size):
        super(LoGCANPlus_Head, self).__init__()


        self.bottleneck1 = conv_3x3(in_channel[0], transform_channel)
        self.bottleneck2 = conv_3x3(in_channel[1], transform_channel)
        self.bottleneck3 = conv_3x3(in_channel[2], transform_channel)
        self.bottleneck4 = conv_3x3(in_channel[3], transform_channel)

        self.decoder_stage1 = nn.Conv2d(transform_channel, num_class, kernel_size=1)
        self.global_gather = SpatialGatherModule()

        self.center1 = RVSA_MRAM(
                dim=transform_channel,
                out_dim=transform_channel,
                num_heads=num_heads,
                patch_size=patch_size,
                num_classes=num_class
            )
        self.center2 = RVSA_MRAM(
                dim=transform_channel,
                out_dim=transform_channel,
                num_heads=num_heads,
                patch_size=patch_size,
                num_classes=num_class
            )
        self.center3 = RVSA_MRAM(
                dim=transform_channel,
                out_dim=transform_channel,
                num_heads=num_heads,
                patch_size=patch_size,
                num_classes=num_class
            )
        self.center4 = RVSA_MRAM(
                dim=transform_channel,
                out_dim=transform_channel,
                num_heads=num_heads,
                patch_size=patch_size,
                num_classes=num_class
            )

        self.catconv1 = conv_3x3(transform_channel * 2, transform_channel)
        self.catconv2 = conv_3x3(transform_channel * 2, transform_channel)
        self.catconv3 = conv_3x3(transform_channel * 2, transform_channel)

        self.catconv = conv_3x3(transform_channel, transform_channel)

    def forward(self, x_list):
        feat1, feat2, feat3, feat4 = self.bottleneck1(x_list[0]), self.bottleneck2(x_list[1]), self.bottleneck3(
            x_list[2]), self.bottleneck4(x_list[3])

        pred1 = self.decoder_stage1(feat4)
        global_center = self.global_gather(feat4, pred1)

        # [b,h,w,k]
        new_feat4 = self.center4(feat4, global_center)

        feat3 = self.catconv1(upsample_add(new_feat4, feat3))
        new_feat3 = self.center3(feat3, global_center)

        feat2 = self.catconv2(upsample_add(new_feat3, feat2))
        new_feat2 = self.center2(feat2, global_center)

        feat1 = self.catconv3(upsample_add(new_feat2, feat1))
        new_feat1 = self.center1(feat1, global_center)

        new_feat4 = F.interpolate(new_feat4, scale_factor=8, mode="bilinear", align_corners=False)
        new_feat3 = F.interpolate(new_feat3, scale_factor=4, mode="bilinear", align_corners=False)
        new_feat2 = F.interpolate(new_feat2, scale_factor=2, mode="bilinear", align_corners=False)

        out = self.catconv(new_feat1 + new_feat2 + new_feat3 + new_feat4)

        return [out, pred1]