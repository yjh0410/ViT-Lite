import torch.nn as nn
from utils.modules import Conv, BottleneckDW, CSPBlockDW, ShuffleBlockv1, ShuffleBlockv2


class RocketNet(nn.Module):
    def __init__(self, model_size='1.0x', num_classes=1000):
        super(RocketNet, self).__init__()
        self.num_classes = num_classes

        # model config
        if model_size == '0.5x':
            cfg = [24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            cfg = [24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            cfg = [24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            cfg = [24, 244, 488, 976, 2048]

        # stride = 4
        self.layer_1 = nn.Sequential(
            Conv(3, cfg[0], k=3, p=1, s=2),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        )
            
        # stride = 8
        self.layer_2 = nn.Sequential(
            ShuffleBlockv2(c1=cfg[0], c2=cfg[1]), # BottleneckDW(cfg[0], cfg[1], s=2, shortcut=False),
            ShuffleBlockv1(c=cfg[1], n=3) # CSPBlockDW(cfg[1], n=3, shortcut=True)
        )
        # stride = 16
        self.layer_3 = nn.Sequential(
            ShuffleBlockv2(c1=cfg[1], c2=cfg[2]), # BottleneckDW(cfg[1], cfg[2], s=2, shortcut=False),
            ShuffleBlockv1(c=cfg[2], n=7) # CSPBlockDW(cfg[2], n=7, shortcut=True)
        )
        # stride = 32
        self.layer_4 = nn.Sequential(
             ShuffleBlockv2(c1=cfg[2], c2=cfg[3]), # BottleneckDW(cfg[2], cfg[3], s=2, shortcut=False),
            ShuffleBlockv1(c=cfg[3], n=3) # CSPBlockDW(cfg[3], n=3, shortcut=True)
        )
        self.conv5 = Conv(cfg[3], cfg[4], k=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cfg[4], self.num_classes)

        self._initialize_weights()


    def _initialize_weights(self):
        print('init weights...')
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.conv5(x)

        x = self.avgpool(x).view(x.size(0), -1)
        x = self.fc(x)

        return x
