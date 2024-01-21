import torch
import torch.nn as nn
import torch.nn.functional as F
from .SWT import SWTForward, SWTInverse
from .LocalRegionTransformer import SelfAttentionBlock
from .blocks import ConvBlock, DeconvBlock, MeanShift, ResBlock
from .asynch import SmiConv, HorConv, VihConv, CConv
from .cross import FusionNet


class WaveletsHighLowProcess(nn.Module):
    def __init__(self, num_features, windows_size, windows_stride, num_head, drop_path):
        super().__init__()
        self.num_features = num_features
        self.WavletDecompose = SWTForward(J=1, mode='zero', wave='haar')
        self.WavletReconstruction = SWTInverse(mode='zero', wave='haar')
        self.High_preprocess = nn.Sequential(
            nn.Conv2d(4 * num_features, num_features, (1, 1), padding=(0, 0), stride=(1, 1), groups=4),
            nn.PReLU(),
            nn.Conv2d(num_features, num_features, (3, 3), padding=(1, 1), stride=(1, 1), groups=4)
        )
        self.HighProcess1 = SelfAttentionBlock(num_features, windows_size[0], windows_stride, num_head[0], drop_path)
        self.HighProcess2 = SelfAttentionBlock(num_features, windows_size[1], windows_stride, num_head[1], drop_path)

        self.High_postprocess = nn.Sequential(
            nn.Conv2d(num_features, 4 * num_features, (1, 1), padding=(0, 0), stride=(1, 1), groups=4),
            nn.PReLU()
        )

        # self.HighExtract41 = ResBlock(num_features, num_features)
        # self.HighExtract42 = nn.Sequential(
        #     nn.Conv2d(num_features, num_features, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
        #     nn.PReLU()
        # )
        self.SmiConv1 = SmiConv(num_features)
        self.HorConv1 = HorConv(num_features)
        self.VihConv1 = VihConv(num_features)
        self.CConv = CConv(num_features)

        # self.SmiConv2 = SmiConv(num_features)
        # self.HorConv2 = HorConv(num_features)
        # self.VihConv2 = VihConv(num_features)
        # self.high_mix = nn.Sequential(
        #     nn.Conv2d(2 * num_features, num_features, 1, 1, 0),
        #     nn.PReLU()
        # )
        # self.low_mix = nn.Sequential(
        #     nn.Conv2d(2 * num_features, num_features, 1, 1, 0),
        #     nn.PReLU()
        # )
        # self.Fusion_high = FusionNet(num_features)
        # self.Fusion_low = FusionNet(num_features)
        self.Fusion = FusionNet(num_features)

    def forward(self, x):
        low, high = self.WavletDecompose(x)
        data = torch.cat([high[0][:, :, 0, ], high[0][:, :, 1, ], high[0][:, :, 2, ], low], dim=1)
        data = self.High_preprocess(data)
        data = self.HighProcess1(data)
        data = self.Fusion(data, data)
        data = self.HighProcess2(data)

        data = self.High_postprocess(data)
        high1, high2, high3, low = torch.split(data, self.num_features, dim=1)
        H1 = self.HorConv1(high1).unsqueeze(2)
        H2 = self.VihConv1(high2).unsqueeze(2)
        H3 = self.SmiConv1(high3).unsqueeze(2)
        # YL = (self.HighExtract42(self.HighExtract41(low)))
        YL = self.CConv(low)
        YH = [torch.cat([H1, H2, H3], dim=2)]
        x = self.WavletReconstruction((YL, YH))
        return x


class DTM(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, windows_size, windows_stride, num_head,
                 drop_path=0.3, upscale_factor=4, act_type='prelu', norm_type=None):
        super().__init__()
        rgb_mean = (0.4679, 0.4481, 0.4029)
        rgb_std = (0.2694, 0.2583, 0.2840)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        self.upscale_factor = upscale_factor
        self.conv_in = ConvBlock(in_channels, 4 * num_features, kernel_size=3, act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4 * num_features, num_features, kernel_size=1, act_type=act_type, norm_type=norm_type)

        self.Tranke = WaveletsHighLowProcess(num_features, windows_size, windows_stride, num_head, drop_path)
        # self.fine = nn.Conv2d(num_features, num_features, 1, padding=0, stride=1)
        if upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7

        self.out = DeconvBlock(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding,
                               act_type='prelu', norm_type=norm_type)
        self.conv_out = ConvBlock(num_features, out_channels, kernel_size=3, act_type=None, norm_type=norm_type)
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        inter_res = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)

        x = self.conv_in(x)
        x = self.feat_in(x)

        x = self.Tranke(x)

        x = self.out(x)
        # x = self.fine(x)

        x = torch.add(inter_res, self.conv_out(x))
        x = self.add_mean(x)
        return x
