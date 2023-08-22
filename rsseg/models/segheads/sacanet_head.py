import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from timm.models.layers import trunc_normal_

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
    patch_h, patch_w = patch_size
    num_h, num_w = H // patch_h, W // patch_w
    out = input.view(B, C, num_h, patch_h, num_w, patch_w)
    out = out.permute(0,2,4,1,3,5).contiguous().view(-1,C,patch_h,patch_w) #(B*num_h*num_w, C, patch_h, patch_w)
    return out

def patch_recover(input, size):
    """
    input: (B*num_h*num_w, C, patch_h, patch_w)
    output: (B, C, H, W)
    """
    N, C, patch_h, patch_w= input.size()
    H, W = size
    num_h, num_w = H // patch_h, W // patch_w
    B = N // (num_h * num_w)

    out = input.view(B, num_h, num_w, C, patch_h, patch_w)
    out = out.permute(0,3,1,4,2,5).contiguous().view(B, C, H, W)
    return out


class Postion_RPE(nn.Module):
    def __init__(self, dim, feature_size = (32, 32)):
        super(Postion_RPE, self).__init__()
        self.f_size = feature_size  # 特征图大小
        self.dim = dim  # 通道数
        self.num_buckets = (2 * self.f_size[0] - 1) * (2 * self.f_size[1] - 1)  # 桶的数量
        self.get_index()
        self.reset_parameters()

    def forward(self, x):
        B = len(x)
        L_query, L_key = self.relative_position_index.shape
        lookup_table = torch.matmul(x, self.loopup_table_weight)
        pos = lookup_table.flatten(1)[:, self.relative_position_index_flatten].view(B, L_query, L_key)
        return pos

    @torch.no_grad()
    def get_index(self):
        coords_h = torch.arange(self.f_size[0])
        coords_w = torch.arange(self.f_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww  # 分别生成x的坐标和y的坐标
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww  分别产生两个存着距离差的矩阵
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.f_size[0] - 1  # shift to start from 0  # 相对位置都加上一个偏移，使其变正
        relative_coords[:, :, 1] += self.f_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.f_size[1] - 1  # 这里的相对位置偏移是从左到右从上到下编码的
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww  # x轴和y轴的偏移相加,第i行第j列代表第i个元素和第j个元素之间的相对距离
        L = self.f_size[0]*self.f_size[1]
        offset = torch.arange(0, L * self.num_buckets, self.num_buckets).view(-1, 1)
        relative_position_index_flatten = (relative_position_index+offset).flatten()
        self.register_buffer("relative_position_index_flatten", relative_position_index_flatten)
        self.register_buffer("relative_position_index", relative_position_index)
        # print(self.relative_position_index.shape)
        # print(self.relative_position_index)
        # print(self.relative_position_index_flatten.shape)
        # print(self.relative_position_index_flatten)

    @torch.no_grad()
    def reset_parameters(self):
        self.loopup_table_weight = nn.Parameter(
            torch.zeros(self.dim, self.num_buckets))
        trunc_normal_(self.loopup_table_weight, std=.02)


"""
self attention block
"""
class SelfAttentionBlock(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs, feature_size):
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
        self.position_encoder = Postion_RPE(dim=transform_channels, feature_size=feature_size)
        self.channel_attention = ChannelAttention(query_in_channels)
    
    def forward(self, query_feats, key_feats, value_feats):
        batch_size = query_feats.size(0)

        query_feats = self.channel_attention(query_feats) * query_feats

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous() #(batch, h*w, c)

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1) #(batch, c, h*w)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous() #(batch, h*w, c)

        sim_map = torch.matmul(query, key)
       
        sim_map = (self.transform_channels ** -0.5) * sim_map

        pos_q = self.position_encoder(query) * (self.transform_channels ** -0.5)

        sim_map += pos_q
        
        sim_map = F.softmax(sim_map, dim=-1) #(batch, h*w, h*w)
        
        context = torch.matmul(sim_map, value) #(batch, h*w, c)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:]) #(batch, c, h, w)

        context = self.out_project(context) #(batch, c, h, w)
        return context


    def buildproject(self, in_channels, out_channels, num_convs):
        convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_convs-1):
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


"""
get class center
input: (b, c, h, w) (b, k, h, w)
output: (b, c, k, 1)
"""
class SpatialGatherModule(nn.Module):
    def __init__(self, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale
    
    def forward(self, features, probs):
        batch_size, num_classes, h, w = probs.size()
        probs = probs.view(batch_size, num_classes, -1) # batch * k * hw
        probs = F.softmax(self.scale * probs, dim=2)

        features = features.view(batch_size, features.size(1), -1)
        features = features.permute(0, 2, 1) # batch * hw * c
        
        ocr_context = torch.matmul(probs, features) #batch * k * c
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3) # batch * c * k * 1

        return ocr_context

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes//16, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


def Recover(feats, preds, context):
    B, C, H, W = feats.size()
    preds = torch.argmax(preds, dim=1) #[B, H, W]
    context = context.squeeze(-1).permute(0, 2, 1).contiguous() #[B, K, C]
    new_context = torch.zeros(B, H*W, C).type_as(feats)

    for batch_idx in range(B):
        context_iter, preds_iter = context[batch_idx], preds[batch_idx] # [K, C] [H, W]
        preds_iter = preds_iter.view(-1)  # [HW]
        new_context[batch_idx] = context_iter[preds_iter]

    new_context = new_context.permute(0, 2, 1).view(B, C, H, W)
    return new_context

class Sacanet(nn.Module):
    def __init__(self, num_class, patch_size=(16, 16)):
        super(Sacanet, self).__init__()
        transform_channel = 128  # sir,this way
        self.patch_size = patch_size
        self.bottleneck = nn.Sequential(
            nn.Conv2d(480, transform_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(transform_channel),
            nn.ReLU(inplace=True)
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(transform_channel, transform_channel//2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(transform_channel//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(transform_channel//2, num_class, kernel_size=1, stride=1, padding=0)
        )

        self.spatial_gather_module = SpatialGatherModule()

        self.correlate_net = SelfAttentionBlock(
            key_in_channels=transform_channel,
            query_in_channels=transform_channel,
            transform_channels=transform_channel//2,
            out_channels=transform_channel,
            key_query_num_convs=2,
            value_out_num_convs=1,
            feature_size=patch_size #这里有个超参数
        )
        self.catconv = nn.Sequential(
            nn.Conv2d(transform_channel*2, transform_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(transform_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_list):

        feats = x_list[-1]
        feats = self.bottleneck(feats)

        preds_stage1 = self.decoder_stage1(feats)
        context = self.spatial_gather_module(feats, preds_stage1) #(b, c, k, 1)
        context = Recover(feats, preds_stage1, context)
        patch_context = patch_split(context, self.patch_size) #(B*num_h*num_w, C, patch_h, patch_w)

        patch_feat = patch_split(feats, self.patch_size) #(B*num_h*num_w, C, patch_h, patch_w)
        patch_pred = patch_split(preds_stage1, self.patch_size) #(B*num_h*num_w, K, patch_h, patch_w)
        patch_local_context = self.spatial_gather_module(patch_feat, patch_pred) #(B*num_h*num_w, c, k, 1)
        patch_local_context = Recover(patch_feat, patch_pred, patch_local_context) #(B*num_h*num_w, C, patch_h, patch_w)

        result = self.correlate_net(patch_feat, patch_local_context, patch_context)
        result = patch_recover(result, (feats.shape[2], feats.shape[3]))
        result = self.catconv(torch.cat([result, feats], dim=1))
        preds_stage2 = result

        #preds_stage1 = F.interpolate(preds_stage1, img_size, mode="bilinear")
        #preds_stage2 = F.interpolate(preds_stage2, img_size, mode="bilinear")


        return [preds_stage2, preds_stage1]



class SelfAttentionBlock2(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs, feature_size):
        super(SelfAttentionBlock2, self).__init__()
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
        self.position_encoder = Postion_RPE(dim=transform_channels, feature_size=feature_size)
        self.channel_attention = ChannelAttention(query_in_channels)

    def forward(self, query_feats):
        key_feats = query_feats
        value_feats = query_feats
        batch_size = query_feats.size(0)

        query_feats = self.channel_attention(query_feats) * query_feats

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()  # (batch, h*w, c)

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1)  # (batch, c, h*w)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()  # (batch, h*w, c)

        sim_map = torch.matmul(query, key)

        sim_map = (self.transform_channels ** -0.5) * sim_map

        pos_q = self.position_encoder(query) * (self.transform_channels ** -0.5)

        sim_map += pos_q

        sim_map = F.softmax(sim_map, dim=-1)  # (batch, h*w, h*w)

        context = torch.matmul(sim_map, value)  # (batch, h*w, c)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])  # (batch, c, h, w)

        context = self.out_project(context)  # (batch, c, h, w)
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


class SelfAttentionBlock3(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs, feature_size):
        super(SelfAttentionBlock3, self).__init__()
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

    def forward(self, query_feats):
        key_feats = query_feats
        value_feats = query_feats
        batch_size = query_feats.size(0)

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()  # (batch, h*w, c)

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1)  # (batch, c, h*w)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()  # (batch, h*w, c)

        sim_map = torch.matmul(query, key)

        sim_map = (self.transform_channels ** -0.5) * sim_map

        sim_map = F.softmax(sim_map, dim=-1)  # (batch, h*w, h*w)

        context = torch.matmul(sim_map, value)  # (batch, h*w, c)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])  # (batch, c, h, w)

        context = self.out_project(context)  # (batch, c, h, w)
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

