"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn import functional as F


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    # embed_dim是卷积核个数，不同的模型给你设置好了不同的embed_dim，这里base模型是768，large和huge也有自己的embed_dim
    # 这边vit-b默认是16倍下采样，即224/16，采完变14
    def __init__(self, img_size=768, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 网格大小，即变成多少个patch
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)  # 再太阳花的博文中写的时候把channel维度放在了宽高后面
        # patch embedding所进行的一系列划分，都可以用一个卷积来达成，用一个卷积核大小为16x16，步距为16，卷积核个数为768的卷积来实现，再经过展平等操作
        # (B,3,224,224)=>(B,768,14,14)=>(B,768,196)=>(B,196,768)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 就是注意力机制中除的那个根号下dk（k的维度）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # 因为Transformer的编码器要求输入的是token即向量类型，下面的c就是输入进编码器一个样本的长度，qkv本质是一个全连接层，经过其之后出来变成3c的长度
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # 因为每个head划分的时候是将得到的qkv进行均分，太阳花讲解中两个head，qkv维度是4，即4的长度变2，也就是用输入的dim/head数，就是C // self.num_heads
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # 因为qkv三者是由同一个全连接层得到的，所以要分别得到qkv就要将他们分开，这边就是对上面得到的qkv进行分开，其实reshape和permute也是为了将qkv进行分开
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PUPHead(nn.Module):
    def __init__(self, num_classes):
        super(PUPHead, self).__init__()

        self.UP_stage_1 = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.UP_stage_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.UP_stage_3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.UP_stage_4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.cls_seg = nn.Conv2d(256, num_classes, 3, padding=1)

    def forward(self, x):
        x = self.UP_stage_1(x)
        x = self.UP_stage_2(x)
        x = self.UP_stage_3(x)
        x = self.UP_stage_4(x)
        x = self.cls_seg(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=768, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

        self.out = Rearrange("b (h w) c->b c h w", h=image_height // patch_height, w=image_width // patch_width)
        # self.Head =PUPHead(num_classes)

    def forward(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)

        selected_layers_output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in [2, 5, 8, 11]:  # 第2，5，8，11层的索引
                # x = self.out(x[:, 1:, :])

                selected_layers_output.append(x)
        result = []

        for x in selected_layers_output:
            x = self.out(x[:, 1:, :])
            result.append(x)

        # x = self.blocks(x)
        # x=self.out(x[:, 1:, :])
        # x=self.Head(x)
        # print(selected_layers_output[0].shape)

        return result
        # x = self.norm(x)
        # if self.dist_token is None:
        #     return self.pre_logits(x[:, 0])
        # else:
        #     return x[:, 0], x[:, 1]

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     if self.head_dist is not None:
    #         x, x_dist = self.head(x[0]), self.head_dist(x[1])
    #         if self.training and not torch.jit.is_scripting():
    #             # during inference, return the average of both classifier predictions
    #             return x, x_dist
    #         else:
    #             return (x + x_dist) / 2
    #     else:
    #         x = self.head(x)
    #     return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(768, 384, kernel_size=1)
        self.conv2 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(384, 192, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Backbone(nn.Module):
    def __init__(self, num_classes):
        super(Backbone, self).__init__()
        self.has_logits = True
        self.backbone = VisionTransformer(img_size=512,
                                          patch_size=16,
                                          embed_dim=768,
                                          depth=12,
                                          num_heads=12,
                                          representation_size=None,
                                          num_classes=num_classes)
        # self.PUPHead = PUPHead(num_classes)
        self.sample = nn.Conv2d(768, 104, kernel_size=3, padding=1)

        self.handle_plus = nn.Conv2d(768, 768, kernel_size=3, padding=1)
        self.half1 = nn.Conv2d(768, 384, kernel_size=3, padding=1)
        self.middle_conv = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.half2 = nn.Conv2d(384, 192, kernel_size=3, padding=1)

        # 第一层
        self.layer1_half1 = nn.Conv2d(768, 384, kernel_size=3, padding=1)
        self.layer1_handle_plus = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer1_middle_conv = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer1_half2 = nn.Conv2d(384, 192, kernel_size=3, padding=1)

        # 第二层
        self.layer2_half1 = nn.Conv2d(768, 384, kernel_size=3, padding=1)
        self.layer2_handle_plus = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer2_middle_conv = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer2_half2 = nn.Conv2d(384, 192, kernel_size=3, padding=1)

        # 第三层
        self.layer3_half1 = nn.Conv2d(768, 384, kernel_size=3, padding=1)
        self.layer3_handle_plus = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer3_middle_conv = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer3_half2 = nn.Conv2d(384, 192, kernel_size=3, padding=1)

        # 第四层
        self.layer4_half1 = nn.Conv2d(768, 384, kernel_size=3, padding=1)
        self.layer4_handle_plus = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer4_middle_conv = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.layer4_half2 = nn.Conv2d(384, 192, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.backbone(x)
        print(x[0].shape,x[1].shape,x[2].shape,x[3].shape)
        # result=[]
        # for x in new_x:
        #     x =self.handle_plus(x)
        #     print(x.shape)
        #     x = self.half1(x)
        #     print(x.shape)
        #     x = self.middle_conv(x)
        #     print(x.shape)
        #     x = self.half2(x)
        #     print(x.shape)
        #     x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)
        #     print('-------')
        #     result.append(x)

        result = []
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

        # networks = [Decoder() for _ in range(len(new_x))]

        # result = [networks[i](new_x[i]) for i in range(len(new_x))]
        for i in range(len(result)):
            result[i] = F.interpolate(result[i], scale_factor=4, mode="bilinear", align_corners=True)

        concat_result = torch.cat(result, dim=1)
        concat_result = self.sample(concat_result)
        concat_result = F.interpolate(concat_result, scale_factor=4, mode="bilinear", align_corners=True)

        # x = self.PUPHead(x)
        return concat_result


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


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
    model = Backbone(num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model


if __name__ == "__main__":
    # VIT-Large  设置了16个patch
    SETRNet = vit_base_patch16_224_in21k(num_classes=104)
    img = torch.randn(1, 3, 512, 512)
    preds = SETRNet(img)
    print(preds.shape)
