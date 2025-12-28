from .swin import SwinT_mod, SwinT
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange


class CTR(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)
        self.scale = nn.Sequential(nn.Linear(dim, 1), nn.GELU())

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        scale = self.scale(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x * scale + _x
        x = x + _x
        return x


class CSA(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)
        self.scale = nn.Sequential(nn.Linear(dim, 1), nn.GELU())

    def patchify(self, lo, hi, lohw, hihw, minhw=7):  # last layer is 7x7
        # unfold feature map into patches
        # hi feat is small, lo feat is large
        # the smallest feature map is 7x7
        B, Nl, Cl = lo.shape
        _, Nh, Ch = hi.shape
        base = hihw // 7
        rate = lohw // hihw * base

        # hi feature
        nphi = rearrange(hi, "b (h w) c -> b c h w", h=hihw, w=hihw)
        nphi = F.unfold(nphi, kernel_size=base, stride=base, padding=0)
        nphi = rearrange(
            nphi, "b (c h w) p -> b p (h w) c", c=Ch, h=base, w=base
        )  # B,Np,N,C

        # lo feature
        nplo = rearrange(lo, "b (h w) c -> b c h w", h=lohw, w=lohw)
        nplo = F.unfold(nplo, kernel_size=rate, stride=rate, padding=0)
        nplo = rearrange(
            nplo, "b (c h w) p -> b p (h w) c", c=Cl, h=rate, w=rate
        )  # B,Np,N,C

        return nplo, nphi

    def forward(self, lo, hi, lohw, hihw, patchify=True):
        # use hi as K,V and lo as Q to generate weighted lo
        # patchify
        if patchify:
            lo, hi = self.patchify(lo, hi, lohw, hihw)  # B,N,C -> B,C,H,W -> B,p,n,C
            B, p, N, C = lo.shape
            base = hihw // 7
        else:
            B, N, C = lo.shape
            base = 1

        _lo = lo
        q = self.c_q(lo)
        k = self.c_k(hi)
        v = self.c_v(hi)
        scale = self.scale(lo)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        if patchify:
            lo = (attn @ v).reshape(B, p, N, C)
        else:
            lo = (attn @ v).reshape(B, N, C)

        lo = self.proj_drop(lo)
        lo = lo * scale + _lo

        if patchify:
            # fold back
            lo = rearrange(
                lo,
                "b p (h w) c -> b (c h w) p",
                c=C,
                h=lohw // hihw * base,
                w=lohw // hihw * base,
            )
            lo = F.fold(
                lo,
                output_size=lohw,
                kernel_size=lohw // hihw * base,
                stride=lohw // hihw * base,
                padding=0,
            )
            lo = rearrange(lo, "b c h w -> b (h w) c", h=lohw, w=lohw)

        return lo


class MSSA(nn.Module):
    def __init__(self, dim, hw_dim, drop=0.1, patchify=False):
        super().__init__()
        self.c_q1 = nn.Linear(dim, dim)
        self.c_k1 = nn.Linear(dim, dim)
        self.c_v1 = nn.Linear(dim, dim)
        self.norm_fact = dim**-0.5
        self.proj_drop1 = nn.Dropout(drop)

        if patchify:
            self.c_q2 = nn.Linear(dim, dim)
            self.c_k2 = nn.Linear(dim, dim)
            self.c_v2 = nn.Linear(dim, dim)
            self.proj_drop2 = nn.Dropout(drop)

        self.softmax = nn.Softmax(dim=-1)
        self.patch_wise = patchify
        self.hw = hw_dim

        self.scale1 = nn.Sequential(nn.Linear(dim, 1), nn.GELU())
        self.scale2 = nn.Sequential(nn.Linear(dim, 1), nn.GELU())

    def patchify(self, feat, channel):
        # unfold feature map into patches
        np = rearrange(feat, "b (h w) c -> b c h w", h=self.hw, w=self.hw)
        np = F.unfold(np, kernel_size=self.hw // 2, stride=self.hw // 2, padding=0)
        np = rearrange(
            np, "b (c h w) p -> b p (h w) c", c=channel, h=self.hw // 2, w=self.hw // 2
        )  # B,Np,N,C
        return np

    def forward(self, x, y):
        # use hi as K,V and lo as Q to generate weighted lo
        B, N, C = x.shape

        _x = x
        q1 = self.c_q1(y)
        k1 = self.c_k1(x)
        v1 = self.c_v1(x)
        # global attn.
        attn_glb = q1 @ k1.transpose(-2, -1) * self.norm_fact
        attn_glb = self.softmax(attn_glb)
        xg = (attn_glb @ v1).reshape(B, N, C)
        xg = self.proj_drop1(xg)
        scale1 = self.scale1(x)
        x = scale1 * xg + _x

        if self.patch_wise:
            _x = x  # B, N, C
            # unfold
            np = self.patchify(x, channel=C)
            npy = self.patchify(y, channel=C)

            q2 = self.c_q2(npy)
            k2 = self.c_k2(np)
            v2 = self.c_v2(np)
            # local attn.
            attn_lcl = q2 @ k2.transpose(-2, -1) * self.norm_fact
            attn_lcl = self.softmax(attn_lcl)
            xl = (attn_lcl @ v2).reshape(B, 4, (self.hw // 2) ** 2, C)
            # fold back
            xl = rearrange(
                xl, "b p (h w) c -> b (c h w) p", c=C, h=self.hw // 2, w=self.hw // 2
            )
            xl = F.fold(
                xl,
                output_size=self.hw,
                kernel_size=self.hw // 2,
                stride=self.hw // 2,
                padding=0,
            )
            xl = rearrange(xl, "b c h w -> b (h w) c", h=self.hw, w=self.hw)

            xl = self.proj_drop2(xl)
            scale2 = self.scale2(x)
            x = scale2 * xl + _x

        return x


class Net(nn.Module):
    def __init__(self, cfg, device):
        super(Net, self).__init__()
        # palce holder no use
        self.device = device
        self.cfg = cfg

        self.encoder = SwinT(cfg, pretrained=True)

        # embed to 768
        self.emb1 = nn.Conv2d(96, 768, kernel_size=1, padding=0, bias=True)
        self.emb2 = nn.Conv2d(192, 768, kernel_size=1, padding=0, bias=True)
        self.emb3 = nn.Conv2d(384, 768, kernel_size=1, padding=0, bias=True)
        self.emb4 = nn.Conv2d(768, 768, kernel_size=1, padding=0, bias=True)

        # channel-wise attention, B,C,N
        self.CA1 = CTR(3136)
        self.CA2 = CTR(784)
        self.CA3 = CTR(196)
        self.CA4 = CTR(49)

        # cross-scale attention, B,N,C
        self.CSA11 = MSSA(dim=768, hw_dim=56, patchify=True)  # CSA(768) #
        self.CSA12 = CSA(768)
        self.CSA13 = CSA(768)
        self.CSA14 = CSA(768)

        self.CSA22 = MSSA(dim=768, hw_dim=28, patchify=True)  # CSA(768) #
        self.CSA23 = CSA(768)
        self.CSA24 = CSA(768)

        self.CSA33 = MSSA(dim=768, hw_dim=14, patchify=True)  # CSA(768) #
        self.CSA34 = CSA(768)

        self.CSA44 = MSSA(dim=768, hw_dim=7, patchify=False)  # CSA(768)

        # norm
        self.ln11 = nn.LayerNorm(768)
        self.ln22 = nn.LayerNorm(768)
        self.ln33 = nn.LayerNorm(768)
        self.ln44 = nn.LayerNorm(768)

        self.ln12 = nn.LayerNorm(768)
        self.ln13 = nn.LayerNorm(768)
        self.ln14 = nn.LayerNorm(768)
        self.ln23 = nn.LayerNorm(768)
        self.ln24 = nn.LayerNorm(768)
        self.ln34 = nn.LayerNorm(768)

        # score generation
        self.fc_score = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 1),
            nn.ReLU(),
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 1),
            nn.Sigmoid(),
        )

        self.seg_embd1 = nn.Embedding(2, 768)  # NR, FR embd
        self.seg_embd2 = nn.Embedding(2, 768)
        self.seg_embd3 = nn.Embedding(2, 768)
        self.seg_embd4 = nn.Embedding(2, 768)

    def common_forward(self, x):
        B, _, _, _ = x.shape

        # in shape of   64,  64,  32,  16,    8
        # in chanl of  128, 128, 256, 512, 1024
        _, layer1, layer2, layer3, layer4 = self.encoder(x)

        # embed to channel 768
        layer1 = self.emb1(layer1)
        layer2 = self.emb2(layer2)
        layer3 = self.emb3(layer3)
        layer4 = self.emb4(layer4)

        # reshape to B,C,N
        layer1 = rearrange(layer1, "b c h w -> b c (h w)", h=56, w=56)  # B,768,3136
        layer2 = rearrange(layer2, "b c h w -> b c (h w)", h=28, w=28)  # B,768,784
        layer3 = rearrange(layer3, "b c h w -> b c (h w)", h=14, w=14)  # B,768,196
        layer4 = rearrange(layer4, "b c h w -> b c (h w)", h=7, w=7)  # B,768,49
        # channel-wise attention
        layer1 = self.CA1(layer1)
        layer2 = self.CA2(layer2)
        layer3 = self.CA3(layer3)
        layer4 = self.CA4(layer4)
        # reshape from B,C,N to B,N,C
        layer1 = rearrange(layer1, "b c (h w) -> b (h w) c", h=56, w=56)  # B,3136,768
        layer2 = rearrange(layer2, "b c (h w) -> b (h w) c", h=28, w=28)  # B,784, 768
        layer3 = rearrange(layer3, "b c (h w) -> b (h w) c", h=14, w=14)  # B,196, 768
        layer4 = rearrange(layer4, "b c (h w) -> b (h w) c", h=7, w=7)  # B,49,  768
        return layer1, layer2, layer3, layer4

    def forward(self, x, y, mode="NR"):
        B = x.shape[0]
        # x is distorted image, y is reference image
        l1, l2, l3, l4 = self.common_forward(x)
        l1y, l2y, l3y, l4y = self.common_forward(y)

        embd1 = self.seg_embd1(torch.zeros(B, 3136).long().to(x.device))
        embd2 = self.seg_embd2(torch.zeros(B, 784).long().to(x.device))
        embd3 = self.seg_embd3(torch.zeros(B, 196).long().to(x.device))
        embd4 = self.seg_embd4(torch.zeros(B, 49).long().to(x.device))
        if mode == "FR":
            embd1y = self.seg_embd1(torch.ones(B, 3136).long().to(x.device))
            embd2y = self.seg_embd2(torch.ones(B, 784).long().to(x.device))
            embd3y = self.seg_embd3(torch.ones(B, 196).long().to(x.device))
            embd4y = self.seg_embd4(torch.ones(B, 49).long().to(x.device))
        else:
            embd1y = self.seg_embd1(torch.zeros(B, 3136).long().to(x.device))
            embd2y = self.seg_embd2(torch.zeros(B, 784).long().to(x.device))
            embd3y = self.seg_embd3(torch.zeros(B, 196).long().to(x.device))
            embd4y = self.seg_embd4(torch.zeros(B, 49).long().to(x.device))

        out11 = self.CSA11(l1 + embd1, l1y + embd1y)
        out22 = self.CSA22(l2 + embd2, l2y + embd2y)
        out33 = self.CSA33(l3 + embd3, l3y + embd3y)
        out44 = self.CSA44(l4 + embd4, l4y + embd4y)

        out12 = self.CSA12(l1, l2, lohw=56, hihw=28, patchify=True)
        out13 = self.CSA13(l1, l3, lohw=56, hihw=14, patchify=True)
        out14 = self.CSA14(l1, l4, lohw=56, hihw=7, patchify=True)

        out23 = self.CSA23(l2, l3, lohw=28, hihw=14, patchify=True)
        out24 = self.CSA24(l2, l4, lohw=28, hihw=7, patchify=True)

        out34 = self.CSA34(l3, l4, lohw=14, hihw=7, patchify=True)

        # aggregate and merge
        out11 = self.ln11(out11)
        out22 = self.ln22(out22)
        out33 = self.ln33(out33)
        out44 = self.ln44(out44)

        out12 = self.ln12(out12)
        out13 = self.ln13(out13)
        out14 = self.ln14(out14)
        out23 = self.ln23(out23)
        out24 = self.ln24(out24)
        out34 = self.ln34(out34)

        out1 = (out11 + out12 + out13 + out14) / 4
        out2 = (out22 + out23 + out24) / 3
        out3 = (out33 + out34) / 2

        out = torch.cat([out1, out2, out3, out44], dim=1)
        # out in shape of B,N,C, which is B,3136,768
        # gen score
        s = self.fc_score(out)
        w = self.fc_weight(out)
        score = torch.sum(s * w, dim=1) / torch.sum(w, dim=1)

        return score
