from torchvision.models.segmentation import deeplabv3_resnet101
import torch
import torch.nn as nn
BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
import math
from timm.models.layers import DropPath
import torch.nn.functional as F
from timm.models.vision_transformer import vit_base_patch16_siglip_512
from timm.models.vision_transformer import vit_base_patch16_siglip_256


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
                .reshape(B, N, 3, self.heads, C // self.heads)
                .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class MultiLabelHead(nn.Module):
    def __init__(
        self,
        num_classes,
        d_encoder,
        hidden_dim,
        n_heads,
        d_ff,
        dropout,
        share_embedding,
        downsample=None,
        mlp=True,
        droppath=0,
    ):
        super().__init__()
        self.share_embedding = share_embedding
        self.mlp = mlp
        self.block = Block(hidden_dim, n_heads, d_ff, dropout, droppath)
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_classes = num_classes
        self.fc = GroupWiseLinear(num_classes, hidden_dim)

        if not share_embedding:
            self.cls_emb = nn.Parameter(torch.randn(1, num_classes, hidden_dim))
            from torch.nn.init import trunc_normal_

            trunc_normal_(self.cls_emb, std=0.02)
        self.scale = hidden_dim ** -0.5

        self.proj_dec = nn.Linear(d_encoder, hidden_dim)
        self.downsample = downsample
        if downsample:
            self.pooling = nn.AdaptiveAvgPool2d(downsample)

    def forward(self, x):
        if self.share_embedding:
            x, cls_emb = x
            cls_emb = cls_emb.unsqueeze(0)
        else:
            cls_emb = self.cls_emb
        if self.downsample:
            x = self.pooling(x)

        B, C = x.size()[:2]
        x = x.view(B, C, -1).permute(0,2,1)
        x = self.proj_dec(x)

        cls_emb = cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        x = self.block(x)
        x = self.norm(x)
        cls_emb = x[:, -self.num_classes :]
        img_pred = self.fc(cls_emb)

        return img_pred

class Bottleneck(nn.Module):
    """Bottleneck module

    Args:
        inplanes (int): no. input channels
        planes (int): no. output channels
        stride (int): stride
        downsample (nn.Module): downsample module
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class RefinementAppNet(nn.Module):
    def __init__(self, n_classes, inputsize, k_size=3, use_bn=False):
        super().__init__()

        self.use_bn = use_bn
        self.w = int(inputsize[0] / 2)
        self.h = int(inputsize[1] / 2)
        # else:
        #     self.w = int(inputsize[0] / 8)
        #     self.h = int(inputsize[1] / 8)
        self.in_index = [0, 1, 2, 3]
        self.input_transform = 'resize_concat'
        if use_bn:
            self.bn0 = BatchNorm2d(n_classes * 2, momentum=BN_MOMENTUM)

        # pixel-aware attention

        # channel attention moudle
        self.adpool = nn.AdaptiveAvgPool2d((int(self.h / 4), int(self.w / 4)))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # 2 conv layers
        self.conv1 = nn.Conv2d(n_classes * 2, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(96, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(96, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # spatial attention moudle
        self.conv_3 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False)
        self.conv_5 = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=(5 - 1) // 2, bias=False)
        self.conv_7 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=(7 - 1) // 2, bias=False)
        self.conv3 = nn.Conv2d(3, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)

        # class-aware attention
        self.class_head = MultiLabelHead(num_classes=n_classes, d_encoder=256, hidden_dim=256, n_heads=4, d_ff=1024,
                                         dropout=0.1, share_embedding=False, downsample=16)
        # 2 residual blocks
        self.residual = self._make_layer(Bottleneck, 96, 64, 2)

        # Prediction head
        self.seg_conv = nn.Conv2d(256, 9, kernel_size=1, stride=1, padding=0, bias=False)
        self.edge_conv = nn.Conv2d(150, 2, kernel_size=1, stride=1, padding=0, bias=False)
        # # edge prediciton
        self.mask_conv = nn.Conv2d(256, n_classes, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False)
        self.mask_conv2 = nn.Conv2d(256, n_classes, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=False)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):

        """Make residual block"""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, fi_segmentation, co_segmentation, aug=False):
        co = F.interpolate(co_segmentation, (self.w, self.h), mode="bilinear", align_corners=False)
        fi = F.interpolate(fi_segmentation, (self.w, self.h), mode="bilinear", align_corners=False)
        x = torch.cat([fi.softmax(1), co.softmax(1)], dim=1)
        if self.use_bn:
            x = self.bn0(x)
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        y = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        s3 = self.conv_3(y)
        s5 = self.conv_5(y)
        s7 = self.conv_7(y)
        s = torch.cat([s3, s5, s7], dim=1)
        out = self.conv3(s)
        out1 = self.avg_pool(x)
        out1 = self.conv(out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = out * out1
        out = self.sigmoid(out)
        x = x * out
        x = self.residual(x)
        class_x = self.class_head(x)
        class_x = self.sigmoid(class_x)
        alter = self.mask_conv(x)
        alter = self.sigmoid(alter)
        sum_weights = (class_x.unsqueeze(2).unsqueeze(3) * alter)
        refine_output = fi * sum_weights + co * (1 - sum_weights)

        if aug is True:
            return fi, refine_output, co
        else:
            return refine_output


class MLA(nn.Module):
    def __init__(self):
        super(MLA, self).__init__()
        self.sample = nn.Conv2d(768, 104, kernel_size=3, padding=1)

        self.layer1_half1 = nn.Conv2d(768, 384, kernel_size=1)
        self.layer1_handle_plus = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer1_middle_conv = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer1_half2 = nn.Conv2d(384, 192, kernel_size=3, padding=1)

        # 第二层
        self.layer2_half1 = nn.Conv2d(768, 384, kernel_size=1)
        self.layer2_handle_plus = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer2_middle_conv = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer2_half2 = nn.Conv2d(384, 192, kernel_size=3, padding=1)

        # 第三层
        self.layer3_half1 = nn.Conv2d(768, 384, kernel_size=1)
        self.layer3_handle_plus = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer3_middle_conv = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer3_half2 = nn.Conv2d(384, 192, kernel_size=3, padding=1)

        # 第四层
        self.layer4_half1 = nn.Conv2d(768, 384, kernel_size=1)
        self.layer4_handle_plus = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer4_middle_conv = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer4_half2 = nn.Conv2d(384, 192, kernel_size=3, padding=1)
    def forward(self, x):
        x[0] = self.layer1_half1(x[0])
        x[1] = self.layer2_half1(x[1])
        x[2] = self.layer3_half1(x[2])
        x[3] = self.layer4_half1(x[3])
        new_x = []
        for i in range(len(x)):
            if i == 0:
                # 数组B的第一个元素就是数组A的第一个元素
                new_x.append(x[i])
            else:
                # 数组B的后续元素是数组A的当前元素与数组B的前一个元素相加
                new_x.append(x[i] + new_x[i - 1])
        result = []
        # 第一层
        new_x[0] = self.layer1_handle_plus(new_x[0])
        new_x[0] = self.layer1_middle_conv(new_x[0])
        new_x[0] = self.layer1_half2(new_x[0])
        result.append(new_x[0])
        # 第二层
        new_x[1] = self.layer2_handle_plus(new_x[1])
        new_x[1] = self.layer2_middle_conv(new_x[1])
        new_x[1] = self.layer2_half2(new_x[1])
        result.append(new_x[1])
        # 第三层
        new_x[2] = self.layer3_handle_plus(new_x[2])
        new_x[2] = self.layer3_middle_conv(new_x[2])
        new_x[2] = self.layer3_half2(new_x[2])
        result.append(new_x[2])
        # 第四层
        new_x[3] = self.layer4_handle_plus(new_x[3])
        new_x[3] = self.layer4_middle_conv(new_x[3])
        new_x[3] = self.layer4_half2(new_x[3])
        result.append(new_x[3])

        for i in range(len(result)):
            result[i] = F.interpolate(result[i], scale_factor=4, mode="bilinear", align_corners=True)

        concat_result = torch.cat(result, dim=1)
        concat_result = self.sample(concat_result)
        concat_result = F.interpolate(concat_result, scale_factor=4, mode="bilinear", align_corners=True)
        return concat_result

class Backbone1(nn.Module):
    def __init__(self):
        super(Backbone1, self).__init__()
        self.Global = vit_base_patch16_siglip_512(pretrained=False)
        self.Local = vit_base_patch16_siglip_256(pretrained=False)
        self.GlobalMLA = MLA()
        self.LocalMLA = MLA()
        self.GLA = RefinementAppNet(n_classes=104,inputsize=[512,512])
    def forward(self, x):
        Global = self.Global(x)

        Global = self.GlobalMLA(Global)
        # Global = Global['out']
        # 将每个张量切分成4份，维度变为(B, 4, C, H/2, W/2)
        B, C, H, W = x.size()
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        new_x = x.view(B * 4, C, H // 2, W // 2)  # 调整形状为 (B*4, C, H/2, W/2)
        # 使用Local模型处理切分后的每一份
        local_out = self.Local(new_x)
        local_out = self.LocalMLA(local_out)
        # 将四个特征图重新组合成一个大的特征图
        _, lc, lh, lw = local_out.size()

        local_out = local_out.view(B, 2, 2, lc, lh, lw)
        local_out = local_out.permute(0, 3, 1, 4, 2, 5).contiguous()
        local_out = local_out.view(B, lc, H, W)
        result = self.GLA(Global,local_out)
        result = F.interpolate(result, scale_factor=2, mode="bilinear", align_corners=False)
        return result



def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    # model = VisionTransformer(img_size=224,
    #                           patch_size=16,
    #                           embed_dim=768,
    #                           depth=12,
    #                           num_heads=12,
    #                           representation_size=768 if has_logits else None,
    #                           num_classes=num_classes)
    model = Backbone1()
    return model






if __name__ == "__main__":
    # VIT-Large  设置了16个patch
    SETRNet = vit_base_patch16_224_in21k(num_classes=104)
    img = torch.randn(2, 3, 512, 512)
    preds = SETRNet(img)
    print(preds.shape)