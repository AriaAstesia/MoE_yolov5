import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()# 3 * 640 * 640
        self.conv1 = Conv(3, 32, 6, 2, 2)# 32 * 160 * 160
        self.conv2 = Conv(32, 64, 3, 2, 1)# 64 * 40 * 40
        ## 输出层
        self.output = nn.Linear(in_features=64 * 40 * 40, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           # [batch, 32,7,7]
        x = x.view(x.size(0), -1)   # 保留batch, 将后面的乘到一起 [batch, 32*7*7]
        output = self.output(x)     # 输出[50,10]
        return output

class LatencyPredictor(nn.Module):
    def __init__(self, feature_dim=12, hidden_dim=400, hidden_layer_num=3):
        super(LatencyPredictor, self).__init__()
        
        self.encoder = Encoder()
        self.first_layer = nn.Linear(feature_dim, hidden_dim)
        self.layers = nn.ModuleList()

        for i in range(hidden_layer_num):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.predict = nn.Linear(hidden_dim, 1)

    def forward(self, x, img):
        
        x = torch.cat((self.encoder(img), x), dim=1)
        
        x = F.relu(self.first_layer(x))

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))

        x = self.predict(x)

        return x