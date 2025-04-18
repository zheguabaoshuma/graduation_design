import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from cv2.gapi import kernel
from einops import rearrange

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 64

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            noise_feature = self.noise_func(noise_embed).view(batch, -1, 1, 1)
            x = x + noise_feature
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Restormer_fn(nn.Module):
    def __init__(self,
                 in_channel=3,
                 out_channel=3,
                 dim=36,
                 num_blocks=[2, 2],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'  ## Other option 'BiasFree'
                 ):

        super(Restormer_fn, self).__init__()

        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(dim),
            nn.Linear(dim, dim * 4),
            Swish(),
            nn.Linear(dim * 4, dim)
        )

        self.patch_embed = OverlapPatchEmbed(in_channel, dim)
        self.patch_embed_refine = OverlapPatchEmbed(in_channel * 2, dim)

        layers = []
        for i in range(num_blocks[0]):
            layers.append(
                TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                 LayerNorm_type=LayerNorm_type))
            layers.append(FeatureWiseAffine(dim, dim))
        self.encoder_level1 = nn.Sequential(*layers)

        layers = []
        for i in range(num_blocks[0]):
            layers.append(
                TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                 LayerNorm_type=LayerNorm_type))
            layers.append(FeatureWiseAffine(dim, dim))
        self.encoder_level1_refine = nn.Sequential(*layers)

        self.reduce_chan_level1_refine = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        layers = []
        for i in range(num_blocks[1]):
            layers.append(
                TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type))
            layers.append(FeatureWiseAffine(dim, dim))
        self.encoder_level2 = nn.Sequential(*layers)

        layers = []
        for i in range(num_blocks[1]):
            layers.append(
                TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type))
            layers.append(FeatureWiseAffine(dim, dim))
        self.encoder_level2_refine = nn.Sequential(*layers)

        self.reduce_chan_level2_refine = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        layers = []
        for i in range(num_blocks[1]):
            layers.append(
                TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type))
            layers.append(FeatureWiseAffine(dim, dim))
        self.decoder_level2 = nn.Sequential(*layers)

        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        layers = []
        for i in range(num_blocks[0]):
            layers.append(
                TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type))
            layers.append(FeatureWiseAffine(dim, dim))
        self.decoder_level1 = nn.Sequential(*layers)

        self.reduce_chan_level_out = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(dim, out_channel, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, vis_ir_img, inp_img, time):
        t = self.noise_level_mlp(time)
        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1_refine = self.patch_embed_refine(vis_ir_img)

        for layer in self.encoder_level1:
            if isinstance(layer, FeatureWiseAffine):
                inp_enc_level1 = layer(inp_enc_level1, t)
            else:
                inp_enc_level1 = layer(inp_enc_level1)
        out_enc_level1 = inp_enc_level1

        for layer in self.encoder_level1_refine:
            if isinstance(layer, FeatureWiseAffine):
                inp_enc_level1_refine = layer(inp_enc_level1_refine, t)
            else:
                inp_enc_level1_refine = layer(inp_enc_level1_refine)
        out_enc_level1_refine = inp_enc_level1_refine

        out_enc_level1 = torch.cat([out_enc_level1, out_enc_level1_refine], 1)
        out_enc_level1 = self.reduce_chan_level1_refine(out_enc_level1)

        inp_enc_level2 = out_enc_level1
        for layer in self.encoder_level2:
            if isinstance(layer, FeatureWiseAffine):
                inp_enc_level2 = layer(inp_enc_level2, t)
            else:
                inp_enc_level2 = layer(inp_enc_level2)
        out_enc_level2 = inp_enc_level2

        inp_enc_level2_refine = out_enc_level1_refine
        for layer in self.encoder_level2_refine:
            if isinstance(layer, FeatureWiseAffine):
                inp_enc_level2_refine = layer(inp_enc_level2_refine, t)
            else:
                inp_enc_level2_refine = layer(inp_enc_level2_refine)
        out_enc_level2_refine = inp_enc_level2_refine

        out_enc_level2 = torch.cat([out_enc_level2, out_enc_level2_refine], 1)
        out_enc_level2 = self.reduce_chan_level2_refine(out_enc_level2)

        inp_dec_level2 = out_enc_level2
        for layer in self.decoder_level2:
            if isinstance(layer, FeatureWiseAffine):
                inp_dec_level2 = layer(inp_dec_level2, t)
            else:
                inp_dec_level2 = layer(inp_dec_level2)
        out_dec_level2 = inp_dec_level2

        inp_dec_level1 = torch.cat([out_dec_level2, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)

        for layer in self.decoder_level1:
            if isinstance(layer, FeatureWiseAffine):
                inp_dec_level1 = layer(inp_dec_level1, t)
            else:
                inp_dec_level1 = layer(inp_dec_level1)
        out_dec_level1 = inp_dec_level1

        out_dec_level1 = torch.cat([out_dec_level1, inp_enc_level1], 1)
        out_dec_level1 = self.reduce_chan_level_out(out_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)
        return out_dec_level1

#############################################################################################
#   WaveUIR
#############################################################################################
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, torch.cat([x_HL, x_LH, x_HH], dim=1)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class DWTBlock(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(DWTBlock, self).__init__()
        self.dwt = DWT()
        self.conv = nn.Conv2d(n_feat, n_feat * 2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        low_bands, high_bands = self.dwt(x)
        return low_bands, high_bands


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class IWTBlock(nn.Module):
    def __init__(self, n_feat, task_num, bias=False):
        super(IWTBlock, self).__init__()
        self.iwt = IWT()
        self.conv_l_iwt = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv_h_iwt = nn.Conv2d(n_feat * 3, n_feat * 3, kernel_size=1, bias=bias)

        # 引入注意力机制
        self.channel_attention = ChannelAttention(n_feat * 4)

        self.out_conv = nn.Conv2d(n_feat, n_feat // 2, kernel_size=1, bias=bias)

    def forward(self, low_bands, high_bands, degradation_weights):
        B, C, H, W = low_bands.shape
        low_bands_iwt = self.conv_l_iwt(low_bands)
        high_bands_iwt = self.conv_h_iwt(high_bands)
        all_bands = torch.cat([low_bands_iwt, high_bands_iwt], dim=1)

        # 应用注意力机制
        attention = self.channel_attention(all_bands)
        all_bands = all_bands * attention
        out = self.iwt(all_bands)

        out = self.out_conv(out)
        return out


class PromptWeightsClassify(nn.Module):
    def __init__(self, task_num=5, lin_dim=192):
        super(PromptWeightsClassify, self).__init__()
        self.conv1 = nn.Conv2d(lin_dim, lin_dim // 4, 1, 1, 0)
        self.conv2 = nn.Conv2d(lin_dim // 4, 64, 1, 1, 0)
        self.linear_layer = nn.Linear(64, task_num)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        x = self.linear_layer(x)
        x = F.softmax(x, dim=1)
        return x


class WaveCoeffFilter(nn.Module):
    def __init__(self, dim, task_num):
        super(WaveCoeffFilter, self).__init__()
        self.dim = dim
        self.threshold_LH = nn.Parameter(torch.ones(1, task_num, dim) * 0.2)
        self.threshold_HL = nn.Parameter(torch.ones(1, task_num, dim) * 0.2)
        self.threshold_HH = nn.Parameter(torch.ones(1, task_num, dim) * 0.2)

    def forward(self, high_bands, degradation_weights):
        B, C, H, W = high_bands.shape
        LH, HL, HH = high_bands.chunk(3, dim=1)
        threshold_LH = F.softmax(self.threshold_LH, dim=2).repeat(B, 1, 1) * degradation_weights.unsqueeze(-1).repeat(1,
                                                                                                                      1,
                                                                                                                      self.dim)
        threshold_LH = torch.sum(threshold_LH, dim=1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)

        threshold_HL = F.softmax(self.threshold_HL, dim=2).repeat(B, 1, 1) * degradation_weights.unsqueeze(-1).repeat(1,
                                                                                                                      1,
                                                                                                                      self.dim)
        threshold_HL = torch.sum(threshold_HL, dim=1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)

        threshold_HH = F.softmax(self.threshold_HH, dim=2).repeat(B, 1, 1) * degradation_weights.unsqueeze(-1).repeat(1,
                                                                                                                      1,
                                                                                                                      self.dim)
        threshold_HH = torch.sum(threshold_HH, dim=1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)

        # 对细节系数进行阈值处理
        LH_thresholded = torch.where(torch.abs(LH) < threshold_LH * torch.max(torch.abs(LH)), torch.zeros_like(LH), LH)
        HL_thresholded = torch.where(torch.abs(HL) < threshold_HL * torch.max(torch.abs(HL)), torch.zeros_like(HL), HL)
        HH_thresholded = torch.where(torch.abs(HH) < threshold_HH * torch.max(torch.abs(HH)), torch.zeros_like(HH), HH)
        high_bands = torch.cat([LH_thresholded, HL_thresholded, HH_thresholded], dim=1)
        return high_bands

class WaveUIR_fn(nn.Module):
    def __init__(self,
        in_channel = 3,
        out_channel = 3,
        dim = 36,
        num_blocks = [4, 6, 6, 8],
        num_refinement_blocks = 4,
        heads = [1, 2, 4, 8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        task_num = 5
    ):
        super(WaveUIR_fn, self).__init__()

        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(dim),
            nn.Linear(dim, dim * 4),
            Swish(),
            nn.Linear(dim * 4, dim)
        )

        self.task_num = task_num

        self.patch_embed = nn.Conv2d(in_channel, dim, kernel_size = 3, stride = 1, padding = 1, bias = bias)
        self.patch_embed_cond = nn.Conv2d(in_channel * 2, dim, kernel_size = 3, stride = 1, padding = 1, bias = bias)

        self.reduce_channel_level1 = nn.Conv2d(int(dim * 2), dim, kernel_size=1)
        self.reduce_channel_level2 = nn.Conv2d(int(2 * dim * 2 ** 1), int(dim * 2 ** 1), kernel_size=1)
        self.reduce_channel_level3 = nn.Conv2d(int(2 * dim * 2 ** 2), int(dim * 2 ** 2), kernel_size=1)
        # self.reduce_channel_level4 = nn.Conv2d(int(2 * dim * 2 ** 3), int(dim * 2 ** 3), kernel_size=1)

        # self.reduce_high_bands_channel_level1 = nn.Conv2d(int(dim * 2), dim, kernel_size=1)
        # self.reduce_high_bands_channel_level1 = nn.Conv2d(int(6 * dim * 2 ** 1), int(3 * dim * 2 ** 1), kernel_size=1)
        # self.reduce_high_bands_channel_level2 = nn.Conv2d(int(6 * dim * 2 ** 2), int(3 * dim * 2 ** 2), kernel_size=1)
        # self.reduce_high_bands_channel_level3 = nn.Conv2d(int(6 * dim * 2 ** 3), int(3 * dim * 2 ** 3), kernel_size=1)

        #####################################################
        #       fusion input
        #####################################################
        encoder_level1_comp = []
        for i in range(num_blocks[0]):
            encoder_level1_comp.append(TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type))
        # encoder_level1_comp.append(FeatureWiseAffine(dim, dim))
        self.encoder_level1 = nn.Sequential(*encoder_level1_comp)

        self.DWTBlock1 = DWTBlock(dim)  ## From Level 1 to Level 2
        encoder_level2_comp = []
        for i in range(num_blocks[1]):
            encoder_level2_comp.append(TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type))
        # encoder_level2_comp.append(FeatureWiseAffine(int(dim * 2 ** 1), int(dim * 2 ** 1)))
        self.encoder_level2 = nn.Sequential(*encoder_level2_comp)

        self.DWTBlock2 = DWTBlock(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        encoder_level3_comp = []
        for i in range(num_blocks[2]):
            encoder_level3_comp.append(TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type))
        # encoder_level3_comp.append(FeatureWiseAffine(int(dim * 2 ** 2), int(dim * 2 ** 2)))
        self.latent = nn.Sequential(*encoder_level3_comp)

        # self.DWTBlock3 = DWTBlock(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        # encoder_level4_comp = []
        # for i in range(num_blocks[3]):
        #     encoder_level4_comp.append(TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type))
        # # encoder_level4_comp.append(FeatureWiseAffine(int(dim * 2 ** 3), int(dim * 2 ** 3)))
        # self.latent = nn.Sequential(*encoder_level4_comp)
        #####################################################
        #       cond input
        #####################################################
        encoder_level1_comp = []
        for i in range(num_blocks[0]):
            encoder_level1_comp.append(TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type))
        # encoder_level1_comp.append(FeatureWiseAffine(dim, dim))
        self.encoder_level1_cond = nn.Sequential(*encoder_level1_comp)

        self.DWTBlock1_cond = DWTBlock(dim)  ## From Level 1 to Level 2
        encoder_level2_comp = []
        for i in range(num_blocks[1]):
            encoder_level2_comp.append(TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type))
        # encoder_level2_comp.append(FeatureWiseAffine(int(dim * 2 ** 1), int(dim * 2 ** 1)))
        self.encoder_level2_cond = nn.Sequential(*encoder_level2_comp)

        self.DWTBlock2_cond = DWTBlock(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        encoder_level3_comp = []
        for i in range(num_blocks[2]):
            encoder_level3_comp.append(TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type))
        # encoder_level3_comp.append(FeatureWiseAffine(int(dim * 2 ** 2), int(dim * 2 ** 2)))
        self.latent_cond = nn.Sequential(*encoder_level3_comp)

        self.DWTBlock3_cond = DWTBlock(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        # encoder_level4_comp = []
        # for i in range(num_blocks[3]):
        #     encoder_level4_comp.append(TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type))
        # # encoder_level4_comp.append(FeatureWiseAffine(int(dim * 2 ** 3), int(dim * 2 ** 3)))
        # self.latent_cond = nn.Sequential(*encoder_level4_comp)

        # Wave Module
        self.Degradation_weights = PromptWeightsClassify(task_num, lin_dim=3 * dim * 2 ** 2)
        self.WaveCoeffFilter3 = WaveCoeffFilter(dim * 2 ** 3, task_num)
        self.WaveCoeffFilter2 = WaveCoeffFilter(dim * 2 ** 2, task_num)
        self.WaveCoeffFilter1 = WaveCoeffFilter(dim * 2 ** 1, task_num)

        self.IWTBlock3 = IWTBlock(int(dim * 2 ** 3), task_num)  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.IWTBlock2 = IWTBlock(int(dim * 2 ** 2), task_num)  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.IWTBlock1 = IWTBlock(int(dim * 2 ** 1),
                                  task_num)  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim * 2), out_channel, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, vis_ir_img, inp_img, time):
        B, _, _, _ = inp_img.shape
        t = self.noise_level_mlp(time)

        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1_cond = self.patch_embed_cond(vis_ir_img)
        for layer in self.encoder_level1:
            if isinstance(layer, FeatureWiseAffine):
                inp_enc_level1 = layer(inp_enc_level1, t)
            else:
                inp_enc_level1 = layer(inp_enc_level1)
        for layer in self.encoder_level1_cond:
            if isinstance(layer, FeatureWiseAffine):
                inp_enc_level1_cond = layer(inp_enc_level1_cond, t)
            else:
                inp_enc_level1_cond = layer(inp_enc_level1_cond)
        out_enc_level1_cond = inp_enc_level1_cond
        out_enc_level1 = inp_enc_level1
        out_enc_level1 = self.reduce_channel_level1(torch.cat([out_enc_level1, out_enc_level1_cond], dim=1))

        inp_enc_level2, high_bands1 = self.DWTBlock1(out_enc_level1)
        inp_enc_level2_cond, high_bands1_cond = self.DWTBlock1_cond(out_enc_level1_cond)
        high_bands1 = high_bands1_cond
        for layer in self.encoder_level2:
            if isinstance(layer, FeatureWiseAffine):
                inp_enc_level2 = layer(inp_enc_level2, t)
            else:
                inp_enc_level2 = layer(inp_enc_level2)
        for layer in self.encoder_level2_cond:
            if isinstance(layer, FeatureWiseAffine):
                inp_enc_level2_cond = layer(inp_enc_level2_cond, t)
            else:
                inp_enc_level2_cond = layer(inp_enc_level2_cond)
        out_enc_level2_cond = inp_enc_level2_cond
        out_enc_level2 = inp_enc_level2
        out_enc_level2 = self.reduce_channel_level2(torch.cat([out_enc_level2, out_enc_level2_cond], dim=1))

        inp_enc_level3, high_bands2 = self.DWTBlock2(out_enc_level2)
        inp_enc_level3_cond, high_bands2_cond = self.DWTBlock2_cond(out_enc_level2_cond)
        high_bands2 = high_bands2_cond
        for layer in self.latent:
            if isinstance(layer, FeatureWiseAffine):
                inp_enc_level3 = layer(inp_enc_level3, t)
            else:
                inp_enc_level3 = layer(inp_enc_level3)
        for layer in self.latent_cond:
            if isinstance(layer, FeatureWiseAffine):
                inp_enc_level3_cond = layer(inp_enc_level3_cond, t)
            else:
                inp_enc_level3_cond = layer(inp_enc_level3_cond)
        out_enc_level3_cond = inp_enc_level3_cond
        out_enc_level3 = inp_enc_level3
        latent = self.reduce_channel_level3(torch.cat([out_enc_level3, out_enc_level3_cond], dim=1))

        # inp_enc_level4, high_bands3 = self.DWTBlock3(out_enc_level3)
        # inp_enc_level4_cond, high_bands3_cond = self.DWTBlock3_cond(out_enc_level3_cond)
        # high_bands3 = high_bands3_cond
        # for layer in self.latent:
        #     if isinstance(layer, FeatureWiseAffine):
        #         inp_enc_level4 = layer(inp_enc_level4, t)
        #     else:
        #         inp_enc_level4 = layer(inp_enc_level4)
        # for layer in self.latent_cond:
        #     if isinstance(layer, FeatureWiseAffine):
        #         inp_enc_level4_cond = layer(inp_enc_level4_cond, t)
        #     else:
        #         inp_enc_level4_cond = layer(inp_enc_level4_cond)
        # out_enc_level4_cond = inp_enc_level4_cond
        # out_enc_level4 = inp_enc_level4
        # latent = self.reduce_channel_level4(torch.cat([out_enc_level4, out_enc_level4_cond], dim=1))

        # Wave module
        if self.task_num != 1:
            degradation_weights = self.Degradation_weights(high_bands2)
        else:
            degradation_weights = torch.ones([B, 1], dtype=torch.float32).cuda()

        high_bands1 = self.WaveCoeffFilter1(high_bands1, degradation_weights)
        high_bands2 = self.WaveCoeffFilter2(high_bands2, degradation_weights)
        # high_bands3 = self.WaveCoeffFilter3(high_bands3, degradation_weights)

        # inp_dec_level3 = self.IWTBlock3(latent, high_bands3, degradation_weights)
        # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.IWTBlock2(latent, high_bands2, degradation_weights)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.IWTBlock1(out_dec_level2, high_bands1, degradation_weights)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1