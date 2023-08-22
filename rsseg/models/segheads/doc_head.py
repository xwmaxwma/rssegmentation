import torch.nn as nn
import torch
import torch.nn.functional as F


def patch_split(input, patch_size):
    """
    input: (B, C, H, W)
    output: (B*num_h*num_w, C, patch_h, patch_w)
    """
    B, C, H, W = input.size()
    patch_h, patch_w = patch_size
    num_h, num_w = H // patch_h, W // patch_w
    out = input.view(B, C, num_h, patch_h, num_w, patch_w)
    out = out.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, patch_h,
                                                          patch_w)  # (B*num_h*num_w, C, patch_h, patch_w)
    return out


def patch_recover(input, img_size):
    """
    input: (B*num_h*num_w, C, patch_h, patch_w)
    output: (B, C, H, W)
    """
    N, C, patch_h, patch_w = input.size()
    H, W = img_size
    num_h, num_w = H // patch_h, W // patch_w
    B = N // (num_h * num_w)

    out = input.view(B, num_h, num_w, C, patch_h, patch_w)
    out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return out


class InnerAttention(nn.Module):
    def __init__(self, feat_size, num_classes, match_size):
        super(InnerAttention, self).__init__()
        self.feat_size = feat_size
        self.match_size = match_size
        self.num_classes = num_classes
        self.feat_norm = nn.LayerNorm(self.feat_size)
        self.q_proj = nn.Linear(self.match_size, self.num_classes)
        self.k_proj = nn.Linear(self.match_size, self.num_classes)
        self.out_patch_corr_proj = nn.Linear(self.num_classes, self.num_classes)

    def forward(self, patch_corr):  # , pos_emb):        # (b',k,16,16)
        b, k, h, w = patch_corr.shape
        patch_corr = patch_corr.reshape(b, -1, k)  # (b, hw, mk)
        q = self.feat_norm(self.q_proj(patch_corr).permute(0, 2, 1))
        k = self.feat_norm(self.k_proj(patch_corr).permute(0, 2, 1))  # (b, k, hw)
        # v = self.feat_norm(patch_corr.permute(0, 2, 1)) #(b,mk,hw)
        v = k

        q = q.permute(0, 2, 1).contiguous()  # (b',hw,k)
        v = v.permute(0, 2, 1).contiguous()  # (b',hw,k)

        attn = (q @ k)  # (b,256,k) * (b,k,256)->(b,256,256)
        attn = F.softmax(attn, dim=-1)

        patch_corr_map = attn @ v  # (b,256,256) * (b,256,k) -> (b,256,k)
        patch_corr_map = patch_corr_map + self.out_patch_corr_proj(patch_corr_map)
        patch_corr_map = patch_corr_map.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return patch_corr_map


class SelfAttentionBlock(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs, num_classes, topk, split_size=(32, 32)):
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
        self.inner_attn = InnerAttention(feat_size=split_size[0] * split_size[1], num_classes=num_classes,
                                         match_size=topk * num_classes)
        self.k = num_classes
        self.topk = topk
        self.split_size = split_size

    def forward(self, query_feats, key_feats, value_feats):
        batch_size, _, h, w = query_feats.shape

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()  # (batch, h*w, c)

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1)  # (batch, c, h*w)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()  # (batch, h*w, c)

        sim_map = torch.matmul(query, key)  # (batch,hw,mk)

        # 加入AiA模块
        corr_map = sim_map.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2).contiguous()  # (b,k,h,w)
        patch_corr = patch_split(corr_map, self.split_size)  # (b*num_h*num_w,k,16,16)
        patch_corr = self.inner_attn(patch_corr)
        corr_map = patch_recover(patch_corr, self.split_size)  # (b,k,h,w)
        corr_map = corr_map.reshape(batch_size, self.k, -1).permute(0, 2, 1).contiguous()
        # sim_map = sim_map + corr_map
        sim_map = corr_map

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


"""
input
x: (batch, c, h, w) 输入特征
preds: (batch, c, h, w) 预测图
output
feats_semantic: (batch, c, topk*k)
"""


class SemanticLevelContext(nn.Module):
    def __init__(self, topk):
        super(SemanticLevelContext, self).__init__()
        self.topk = topk

    def forward(self, x, preds):
        inputs = x
        batch_size, num_channels, h, w = x.size()
        num_classes = preds.size(1)

        feats_semantic = torch.zeros(batch_size, self.topk * num_classes, num_channels).type_as(x)  # (b, 5k, c)

        for batch_idx in range(batch_size):
            feats_iter, preds_iter = x[batch_idx], preds[batch_idx]
            feats_iter, preds_iter = feats_iter.reshape(num_channels, -1), preds_iter.reshape(num_classes, -1)
            feats_iter, preds_iter = feats_iter.permute(1, 0), preds_iter.permute(1, 0)
            argmax = preds_iter.argmax(1)

            for clsid in range(num_classes):
                mask = (argmax == clsid)
                if mask.sum() == 0: continue
                feats_iter_cls = feats_iter[mask]  # (m, c)
                # preds_iter_clss = preds_iter[:, clsid][mask]
                preds_iter_cls = preds_iter[:, :][mask]  # (m, k)
                m = preds_iter_cls.size(0)

                top2_scores = torch.topk(preds_iter_cls, k=2, dim=1)[0]
                certainty = (top2_scores[:, 0] - top2_scores[:, 1])  # (m)
                weight = F.softmax(certainty, dim=0)  # (m)

                if m >= self.topk:
                    for t in range(1, self.topk + 1):
                        certainty_low = torch.topk(certainty, k=m * t // self.topk, dim=0, largest=True)[0]
                        certainty_low = certainty_low[-1]

                        weight_mask = (certainty >= certainty_low)
                        feat_highscore_cls = feats_iter_cls[weight_mask]  # (topm, c)

                        new_certainty = certainty[weight_mask]
                        new_weight = F.softmax(new_certainty, dim=0)

                        feats_iter_clss = feat_highscore_cls * new_weight.unsqueeze(-1)
                        feats_iter_clss = feats_iter_clss.sum(0)
                        feats_semantic[batch_idx][clsid * self.topk + t - 1] = feats_iter_clss
                else:
                    # weight = F.softmax(preds_iter_clss, dim=0)
                    # feats_iter_cls = feats_iter_cls * weight.unsqueeze(-1)
                    feats_iter_cls = feats_iter_cls * weight.unsqueeze(-1)
                    feats_iter_cls = feats_iter_cls.sum(0)
                    for t in range(1, self.topk + 1):
                        feats_semantic[batch_idx][clsid * self.topk + t - 1] = feats_iter_cls

        feats_semantic = feats_semantic.permute(0, 2, 1).contiguous().unsqueeze(-1)  # (batch, c, topk*k, 1)
        feats_semantic_global = feats_semantic.reshape(feats_semantic.size(0), feats_semantic.size(1), self.topk,
                                                       -1).permute(0, 1, 3, 2).contiguous()[:, :, :, -1].unsqueeze(
            -1)  # (b, c, k, topk) 这里可选择全局类是用前多少的，0为显著特征，-1为平均类中心
        # print(feats_semantic.shape, feats_semantic_global.shape)

        return feats_semantic, feats_semantic_global


class ObjectContextBlock(SelfAttentionBlock):
    def __init__(self, in_channels, transform_channels, num_classes, topk):
        super(ObjectContextBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            transform_channels=transform_channels,
            out_channels=in_channels,
            key_query_num_convs=2,
            value_out_num_convs=1,
            num_classes=num_classes,
            topk=topk
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, query_feats, key_feats, value_feats):
        h, w = query_feats.size()[2:]
        context = super(ObjectContextBlock, self).forward(query_feats, key_feats,
                                                          value_feats)  # (batch, c, h, w)(batch, c, k, 1)->(batch,c,h,w)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        return output

class DOC_Head(nn.Module):
    def __init__(self, num_class):
        super(DOC_Head, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(480, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # auxiliary
        self.auxiliary_decoder = nn.Sequential(
            nn.Conv2d(480, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_class, kernel_size=1, stride=1, padding=0)
        )
        self.topk=1
        # self.spatial_gather_module = SpatialGatherModule()
        self.spatial_gather_module = SemanticLevelContext(self.topk)
        self.object_context_block = ObjectContextBlock(
            in_channels=512,
            transform_channels=256,
            num_classes=num_class,
            topk=self.topk
        )

    def forward(self, backbone_outputs):
        feats = self.bottleneck(backbone_outputs[-1])

        auxiliary_feats = backbone_outputs[-1]
        preds_auxiliary = self.auxiliary_decoder(auxiliary_feats)

        context, context_global = self.spatial_gather_module(feats, preds_auxiliary)  # (batch, c, k, 1)
        feats = self.object_context_block(feats, context, context_global)

        return [feats,preds_auxiliary]
