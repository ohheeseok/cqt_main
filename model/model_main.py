import copy
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
from .inceptionresnetv2 import inceptionresnetv2
from base.base_model import BaseModel
from torch import nn, einsum
from einops import rearrange, reduce


'''quality encoder E_q'''
class Encoder(nn.Module):
    def __init__(self, ngf, d_model=256):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(3, ngf, 1, 1, 0),
            ResnetBlock(ngf * 1, ngf * 1, 3, 1, 1, num_res=1), DwSample(ngf * 1),
            ResnetBlock(ngf * 1, ngf * 2, 3, 1, 1, num_res=1), DwSample(ngf * 2),
            ResnetBlock(ngf * 2, ngf * 4, 3, 1, 1, num_res=1), DwSample(ngf * 4),
            ResnetBlock(ngf * 4, ngf * 4, 3, 1, 1, num_res=1),
            nn.GroupNorm(num_groups=32, num_channels=ngf * 4, eps=1e-6, affine=True),
            nn.Conv2d(ngf * 4, d_model, 3, 1, 1)
        )

    def forward(self, x):
        return self.op(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, num_res):
        super().__init__()
        self.num_res = num_res
        for idx in range(self.num_res):
            self.add_module("norm_head_{}".format(idx),
                            nn.GroupNorm(num_groups=32, num_channels=in_dim, eps=1e-6, affine=True))
            self.add_module("norm_tail_{}".format(idx),
                            nn.GroupNorm(num_groups=32, num_channels=out_dim, eps=1e-6, affine=True))
            self.add_module("op_head_{}".format(idx), nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding))
            self.add_module("op_tail_{}".format(idx), nn.Conv2d(out_dim, out_dim, kernel_size, stride, padding))
            self.add_module("short_cut_{}".format(idx), nn.Conv2d(in_dim, out_dim, 3, 1, 1))
            in_dim = out_dim

    def forward(self, x):
        for idx in range(self.num_res):
            h = x
            h = getattr(self, "norm_head_{}".format(idx))(h)
            h = h * torch.sigmoid(h)  # swish
            h = getattr(self, "op_head_{}".format(idx))(h)
            h = getattr(self, "norm_tail_{}".format(idx))(h)
            h = h * torch.sigmoid(h)
            h = getattr(self, "op_tail_{}".format(idx))(h)
            x = h + getattr(self, "short_cut_{}".format(idx))(x)
        return x

class DwSample(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.op = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.op(x)


'''CQT(convolved quality transformer)'''
class LQPU(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.depthwise = nn.Conv2d(n_in, n_in, kernel_size=3, padding=1, groups=n_in)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        x = self.depthwise(x) + self.shortcut(x)
        return x

class InvResFFN(nn.Module):
    def __init__(self, n_in, expand=4):
        super().__init__()
        self.conv1 = nn.Conv2d(n_in, n_in * expand, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(n_in * expand)
        self.depthwise = nn.Conv2d(n_in * expand, n_in * expand, kernel_size=3, padding=1, groups=n_in * expand)
        self.bn2 = nn.BatchNorm2d(n_in * expand)
        self.conv3 = nn.Conv2d(n_in * expand, n_in, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(n_in)
        self.shortcut = nn.Sequential()
        self.gelu = nn.GELU()

    def forward(self, x):
        x_ = x
        x = self.gelu(self.bn1(self.conv1(x)))
        x = self.gelu(self.bn2(self.depthwise(x)))
        x = self.bn3(self.conv3(x)) + self.shortcut(x_)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=128, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CQTBlock(nn.Module):  # CQT block
    def __init__(self, h, w, dim=512, heads=4, dim_head=128, dropout=0.):
        super().__init__()
        self.h = h
        self.w = w
        self.num_token = h * w
        self.lpu = LQPU(dim)
        self.shortcut = nn.Sequential()
        self.ln1 = nn.LayerNorm(dim, eps=1e-5)
        self.mhsa = Attention(dim, heads, dim_head, dropout)
        self.ln2 = nn.LayerNorm(dim, eps=1e-5)
        self.irffn = InvResFFN(dim)

    def forward(self, x):
        x = self.lpu(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        x_ = x
        x = self.ln1(x)
        x = self.mhsa(x) + self.shortcut(x_)

        x_ = x
        x = self.ln2(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.h, w=self.w)
        x = self.irffn(x)

        return x

class MlpHead(nn.Module):
    def __init__(self, d_model=3072, dim_mlp=2048, dropout=0.1):
        super().__init__()
        self.op = nn.Sequential(
            nn.Linear(d_model, dim_mlp), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_mlp, 1),
        )

    def forward(self, x):
        return self.op(x)


'''Main model'''
class Model(BaseModel):
    def __init__(self, ngf=64, h=32, w=32, dim=512, heads=4, dim_head=128, dropout=0.5, dim_mlp=2048):
        super().__init__()
        self.h = h
        self.w = w
        self.num_token = h * w

        self.quality_encoder = Encoder(ngf, dim // 2)
        self.semantic_normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.semantic_encoder = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        self.conv_1x1 = nn.Conv2d(1920, dim // 2, (1, 1))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_token, dim))

        self.cqt_stage1_layer1 = CQTBlock(h, w, dim, heads, dim_head, dropout=0.)
        self.cqt_stage1_layer2 = CQTBlock(h, w, dim, heads, dim_head, dropout=0.)
        self.downsample = DwSample(dim, out_dim=dim * 2)
        self.cqt_stage2_layer1 = CQTBlock(h // 2, w // 2, dim * 2, heads, dim_head * 2, dropout=0.)
        self.cqt_stage2_layer2 = CQTBlock(h // 2, w // 2, dim * 2, heads, dim_head * 2, dropout=0.)
        self.mlp = MlpHead(d_model=dim * 6, dim_mlp=dim_mlp, dropout=dropout)

        self.loss_reg = nn.L1Loss()

    def forward(self, img_ref, img_dist):

        '''Encoding'''
        feat_q_ref = self.quality_encoder(img_ref)
        feat_q_dist = self.quality_encoder(img_dist)
        self.semantic_encoder.eval()
        with torch.no_grad():
            _, feat_s_ref = self.semantic_encoder(self.semantic_normalize(img_ref))
            feat_s_ref = F.interpolate(feat_s_ref.detach(), size=(32, 32), mode='bilinear', align_corners=True)
            _, feat_s_dist = self.semantic_encoder(self.semantic_normalize(img_dist))
            feat_s_dist = F.interpolate(feat_s_dist.detach(), size=(32, 32), mode='bilinear', align_corners=True)
        feat_s_diff = torch.subtract(feat_s_ref, feat_s_dist)
        feat_s_diff = self.conv_1x1(feat_s_diff)
        feat_q_diff = torch.subtract(feat_q_ref, feat_q_dist)

        '''CQT layers'''
        feat_q_diff_seq = rearrange(feat_q_diff, 'b c h w -> b (h w) c')
        feat_s_diff_seq = rearrange(feat_s_diff, 'b c h w -> b (h w) c')
        feat_t_diff_seq = torch.cat([feat_s_diff_seq, feat_q_diff_seq], dim=2) + self.pos_embed[0, :self.num_token]
        feat_t_diff = rearrange(feat_t_diff_seq, 'b (h w) c -> b c h w', h=self.h, w=self.w)
        feat_1 = self.cqt_stage1_layer1(feat_t_diff)
        feat_1 = self.cqt_stage1_layer2(feat_1)
        feat_2 = self.downsample(feat_1)
        feat_2 = self.cqt_stage2_layer1(feat_2)
        feat_2 = self.cqt_stage2_layer2(feat_2)
        feat_1_seq = rearrange(feat_1, 'b c h w -> b (h w) c')
        feat_2_seq = rearrange(feat_2, 'b c h w -> b (h w) c')
        feat_1_mean = feat_1_seq.mean(dim=1)
        feat_2_mean = feat_2_seq.mean(dim=1)
        feat_1_p, _ = torch.topk(feat_1_seq, k=int(self.num_token * 0.05), dim=1)   # p-percentile pooling (5%)
        feat_1_p_mean = feat_1_p.mean(dim=1)
        feat_2_p, _ = torch.topk(feat_2_seq, k=int(self.num_token * 0.05), dim=1)   # p-percentile pooling (5%)
        feat_2_p_mean = feat_2_p.mean(dim=1)
        feat_merge = torch.cat([feat_1_mean, feat_1_p_mean, feat_2_mean, feat_2_p_mean], dim=1)
        pred = self.mlp(feat_merge.squeeze())
        return pred

    def backward(self, img_ref, img_dist, gt, optimizer=None):
        self.logs = {}
        pred = self.forward(img_ref, img_dist)
        loss_reg = self.loss_reg(pred, gt.reshape(-1, 1))

        if optimizer is not None:
            optimizer.zero_grad()
            loss_reg.backward()
            optimizer.step()

        self.logs.update({'loss_reg': loss_reg,
                          'pred': pred})
        return pred, self.logs

if __name__ == '__main__':
    img_ref = torch.rand(8, 3, 256, 256)
    img_dist = torch.rand(8, 3, 256, 256)

    model = Model(ngf=64, h=32, w=32, dim=512, heads=4, dim_head=128, dropout=0.1, dim_mlp=2048)
    pred = model(img_ref, img_dist)
    print(pred.shape)