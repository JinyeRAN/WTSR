import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionNet(nn.Module):
    def __init__(self, num_features):
        super(FusionNet, self).__init__()
        self.ConvMix = nn.Sequential(
            nn.Conv2d(num_features*2, num_features, kernel_size=1, padding=0, stride=1),
            nn.PReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, stride=1)
        )

    def bis(self, input, dim, index):
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, x8, x4):
        x4_unfold = F.unfold(x4, kernel_size=(3, 3), padding=1, stride=3)
        x8_unfold = F.unfold(x8, kernel_size=(3, 3), padding=1, stride=3)
        x8x_unfold = x8_unfold.permute(0, 2, 1)

        x4_unfold = F.normalize(x4_unfold, dim=1)
        x8x_unfold = F.normalize(x8x_unfold, dim=2)

        vector = torch.bmm(x8x_unfold, x4_unfold)
        v1, idx1 = torch.sort(vector, dim=2, descending=True)

        id2 = idx1[:, :, 1]
        transfer = self.bis(x4_unfold, dim=2, index=id2)
        tmpid2 = F.fold(transfer, output_size=x8.size()[-2:], kernel_size=(3, 3), padding=1, stride=3)
        x = self.ConvMix(torch.cat([x4, tmpid2], dim=1))
        return x


# data = torch.randn(16,48,80,80)
# model = MultiScaleFusionNet(48)
# output = model(data,data)
# print('Done')