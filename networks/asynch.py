import torch
import torch.nn as nn
from .atten import CoordAtt


class SemiConv(nn.Module):
    def __init__(self, num_features, id):
        super(SemiConv, self).__init__()
        if id:
            self.Conv1 = nn.Conv2d(num_features*2, num_features, 1, padding=(0, 0), stride=(1, 1))
            self.Conv2 = nn.Conv2d(num_features*2, num_features, 1, padding=(0, 0), stride=(1, 1))
            self.Conv3 = nn.Conv2d(num_features*2, num_features, 1, padding=(0, 0), stride=(1, 1))
        else:
            self.Conv1 = nn.Conv2d(num_features, num_features, 1, padding=(0, 0), stride=(1, 1))
            self.Conv2 = nn.Conv2d(num_features, num_features, 1, padding=(0, 0), stride=(1, 1))
            self.Conv3 = nn.Conv2d(num_features, num_features, 1, padding=(0, 0), stride=(1, 1))
        self.act = nn.PReLU()
        self.pad = nn.ZeroPad2d(padding=(1, 1, 1, 1))

    def forward(self, x):
        xt1 = self.Conv1(x)
        x1 = self.pad(x)
        x2 = x1[:, :, :-2, 2:]
        xt2 = self.Conv2(x2)
        x3 = x1[:, :, 2:, :-2]
        xt3 = self.Conv3(x3)
        x = (xt1 + xt2 +xt3)/3
        return self.act(x)


class SmiConv(nn.Module):
    def __init__(self, num_features):
        super(SmiConv, self).__init__()
        self.ConvS1 = SemiConv(num_features,0)
        self.act1 = nn.PReLU()

        self.ConvD = nn.Conv2d(num_features * 2, num_features * 2, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.act2 = nn.PReLU()

        self.ConvS2 = SemiConv(num_features,1)

        self.atten = CoordAtt(num_features * 2, num_features * 2)

    def forward(self, x):
        x_ = self.ConvS1(x)
        x_ = self.act1(x_)
        x_ = torch.cat([x, x_], dim=1)

        x_ = self.ConvD(x_)
        x_ = self.atten(x_)
        x_ = self.act2(x_)

        x_ = self.ConvS2(x_)
        x_ = x_ + x

        return x_


class VihConv(nn.Module):
    def __init__(self, num_features):
        super(VihConv, self).__init__()
        self.ConvV1 = nn.Conv2d(num_features, num_features, kernel_size=(1, 3), padding=(0, 1), stride=(1, 1))
        self.act1 = nn.PReLU()

        self.ConvD = nn.Conv2d(num_features * 2, num_features * 2, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.act2 = nn.PReLU()

        self.ConvV2 = nn.Conv2d(num_features * 2, num_features, kernel_size=(1, 3), padding=(0, 1), stride=(1, 1))

        self.atten = CoordAtt(num_features * 2, num_features * 2)

    def forward(self, x):
        x_ = self.ConvV1(x)
        x_ = self.act1(x_)
        x_ = torch.cat([x, x_], dim=1)

        x_ = self.ConvD(x_)
        x_ = self.atten(x_)
        x_ = self.act2(x_)

        x_ = self.ConvV2(x_)
        x_ = x_ + x

        return x_


class HorConv(nn.Module):
    def __init__(self, num_features):
        super(HorConv, self).__init__()
        self.ConvH1 = nn.Conv2d(num_features, num_features, kernel_size=(3, 1), padding=(1, 0), stride=(1, 1))
        self.act1 = nn.PReLU()

        self.ConvD = nn.Conv2d(num_features * 2, num_features * 2, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.act2 = nn.PReLU()

        self.ConvH2 = nn.Conv2d(num_features * 2, num_features, kernel_size=(3, 1), padding=(1, 0), stride=(1, 1))

        self.atten = CoordAtt(num_features * 2, num_features * 2)

    def forward(self, x):
        x_ = self.ConvH1(x)
        x_ = self.act1(x_)
        x_ = torch.cat([x, x_], dim=1)

        x_ = self.ConvD(x_)
        x_ = self.atten(x_)
        x_ = self.act2(x_)

        x_ = self.ConvH2(x_)
        x_ = x_ + x

        return x_


class CConv(nn.Module):
    def __init__(self, num_features):
        super(CConv, self).__init__()
        self.ConvH1 = nn.Conv2d(num_features, num_features, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.act1 = nn.PReLU()

        self.ConvD = nn.Conv2d(num_features * 2, num_features * 2, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1))
        self.act2 = nn.PReLU()

        self.ConvH2 = nn.Conv2d(num_features * 2, num_features, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

        self.atten = CoordAtt(num_features * 2, num_features * 2)

    def forward(self, x):
        x_ = self.ConvH1(x)
        x_ = self.act1(x_)
        x_ = torch.cat([x, x_], dim=1)

        x_ = self.ConvD(x_)
        x_ = self.atten(x_)
        x_ = self.act2(x_)

        x_ = self.ConvH2(x_)
        x_ = x_ + x

        return x_