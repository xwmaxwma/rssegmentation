import torch.nn as nn
import torch
import torch.nn.functional as F

def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )
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


class SelfAttentionBlock(nn.Module):
    """
    query_feats: (B*num_h*num_w, C, patch_h, patch_w)
    key_feats: (B*num_h*num_w, C, K, 1)
    value_feats: (B*num_h*num_w, C, K, 1)

    output: (B*num_h*num_w, C, patch_h, patch_w)
    """

    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs):
        super(SelfAttentionBlock, self).__init__()
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
        )
        self.query_project = self.buildproject(
            in_channels=query_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs
        )
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=value_out_num_convs
        )
        self.out_project = self.buildproject(
            in_channels=transform_channels,
            out_channels=out_channels,
            num_convs=value_out_num_convs
        )
        self.transform_channels = transform_channels

    def forward(self, query_feats, key_feats, value_feats):
        batch_size = query_feats.size(0)

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()  # (B*num_h*num_w, patch_h*patch_w, C)

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1)  # (B*num_h*num_w, C, K)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()  # (B*num_h*num_w, K, C)

        sim_map = torch.matmul(query, key)

        sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  # (B*num_h*num_w, patch_h*patch_w, K)

        context = torch.matmul(sim_map, value)  # (B*num_h*num_w, patch_h*patch_w, C)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])  # (B*num_h*num_w, C, patch_h, patch_w)

        context = self.out_project(context)  # (B*num_h*num_w, C, patch_h, patch_w)
        return context

    def buildproject(self, in_channels, out_channels, num_convs):
        convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_convs - 1):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        if len(convs) > 1:
            return nn.Sequential(*convs)
        return convs[0]


class SpatialGatherModule(nn.Module):
    def __init__(self, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, features, probs):
        batch_size, num_classes, h, w = probs.size()
        probs = probs.view(batch_size, num_classes, -1)  # batch * k * hw
        probs = F.softmax(self.scale * probs, dim=2)

        features = features.view(batch_size, features.size(1), -1)
        features = features.permute(0, 2, 1)  # batch * hw * c

        ocr_context = torch.matmul(probs, features)  # (B, k, c)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(-1)  # (B, C, K, 1)
        return ocr_context

class MRAM(nn.Module):
    """
    feat: (B, C, H, W)
    global_center: (B, C, K, 1)
    """

    def __init__(self, in_channels, inner_channels, num_class, patch_size=(4, 4)):
        super(MRAM, self).__init__()
        self.patch_size = patch_size
        self.feat_decoder = nn.Conv2d(in_channels, num_class, kernel_size=1)

        self.correlate_net = SelfAttentionBlock(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            transform_channels=inner_channels,
            out_channels=in_channels,
            key_query_num_convs=2,
            value_out_num_convs=1
        )

        self.get_center = SpatialGatherModule()

        self.cat_conv = nn.Sequential(
            conv_3x3(in_channels * 2, in_channels),
            nn.Dropout2d(0.1),
            conv_3x3(in_channels, in_channels),
            nn.Dropout2d(0.1)
        )

    def forward(self, feat, global_center):
        pred = self.feat_decoder(feat)
        patch_feat = patch_split(feat, self.patch_size)  # (B*num_h*num_w, C, patch_h, patch_w)
        patch_pred = patch_split(pred, self.patch_size)  # (B*num_h*num_w, K, patch_h, patch_w)
        local_center = self.get_center(patch_feat, patch_pred)  # (B*num_h*num_w, C, K, 1)
        num_h, num_w = self.patch_size
        global_center = global_center.repeat(num_h * num_w, 1, 1, 1)

        new_feat = self.correlate_net(patch_feat, local_center, global_center)  # (B*num_h*num_w, C, patch_h, patch_w)
        new_feat = patch_recover(new_feat, self.patch_size)  # (B, C, H, W)
        out = self.cat_conv(torch.cat([feat, new_feat], dim=1))

        return out

class SpatialGatherModule(nn.Module):
    def __init__(self, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, features, probs):
        batch_size, num_classes, h, w = probs.size()
        probs = probs.view(batch_size, num_classes, -1)  # batch * k * hw
        probs = F.softmax(self.scale * probs, dim=2)

        features = features.view(batch_size, features.size(1), -1)
        features = features.permute(0, 2, 1)  # batch * hw * c

        ocr_context = torch.matmul(probs, features)  # (B, k, c)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(-1)  # (B, C, K, 1)
        return ocr_context

def upsample_add(x_small, x_big):
    x_small = F.interpolate(x_small, scale_factor=2, mode="bilinear", align_corners=False)
    return torch.cat([x_small, x_big], dim=1)

class LoGCAN_Head(nn.Module):
    def __init__(self, in_channel=[256, 512, 1024, 2048], transform_channel=128, num_class=6):
        super(LoGCAN_Head, self).__init__()
        self.bottleneck1 = conv_3x3(in_channel[0], transform_channel)
        self.bottleneck2 = conv_3x3(in_channel[1], transform_channel)
        self.bottleneck3 = conv_3x3(in_channel[2], transform_channel)
        self.bottleneck4 = conv_3x3(in_channel[3], transform_channel)

        self.decoder_stage1 = nn.Conv2d(transform_channel, num_class, kernel_size=1)
        self.global_gather = SpatialGatherModule()

        self.center1 = MRAM(transform_channel, transform_channel//2, num_class)
        self.center2 = MRAM(transform_channel, transform_channel//2, num_class)
        self.center3 = MRAM(transform_channel, transform_channel//2, num_class)
        self.center4 = MRAM(transform_channel, transform_channel//2, num_class)

        self.catconv1 = conv_3x3(transform_channel*2, transform_channel)
        self.catconv2 = conv_3x3(transform_channel*2, transform_channel)
        self.catconv3 = conv_3x3(transform_channel*2, transform_channel)

        self.catconv = conv_3x3(transform_channel, transform_channel)

    def forward(self, x_list):
        feat1, feat2, feat3, feat4 = self.bottleneck1(x_list[0]), self.bottleneck2(x_list[1]), self.bottleneck3(x_list[2]), self.bottleneck4(x_list[3])
        pred1 =  self.decoder_stage1(feat4)

        global_center = self.global_gather(feat4, pred1)

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