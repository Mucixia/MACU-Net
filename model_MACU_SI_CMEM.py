import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np 
import os
from torch_radon import RadonFanbeam


#CAB block
class CAB(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, H, W) --> (B, C, 1, 1)
        self.conv1x1 = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        weights = self.fc(y).view(b, c, 1, 1)
        x_weighted = x * weights
        x_important = self.conv1x1(x_weighted)


        return x_important

# define basic block of MACU-Net
class  BasicBlock(nn.Module):

    def __init__(self, features):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(features, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(features, features, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(features, features, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, features, 3, 3)))

        self.cab = CAB(features)

        self.det_count = 1024 
        self.source_distance = 500
        self.det_spacing = 2

    def forward(self, x, z,  prev_features, theta, sinogram):
        """
        x:              image from last stage, (batch_size, channel, height, width)
        theta:          angle vector for radon/iradon transform
        sinogram:       measured signogram of the target image
        lambda_step:    gradient descent step
        soft_thr:       soft-thresholding value
        prev_features:  accumulated dense features from previous stages。0、32、64、96...channels
        z:              c-1 features
        """
        radon = RadonFanbeam(x.shape[2], theta,
                             source_distance=self.source_distance,
                             det_distance=self.source_distance, det_count=self.det_count,
                             det_spacing=self.det_spacing)
        sino_pred = radon.forward(x)
        filtered_sinogram = radon.filter_sinogram(sino_pred)
        X_fbp = radon.backprojection(filtered_sinogram - sinogram)#
        x_input = x - self.lambda_step * X_fbp

        x_input = torch.cat([x_input,z],1) # SI concat

        x_input2 = torch.cat([x_input, prev_features], 1)  # CMEM concat


        x = self.cab(x_input2)
        
        x = F.conv2d(x, self.conv1_forward, padding=1)
        x = F.relu(x)
        
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)
        

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        
        x_pred = x_backward 

        return x_pred, x_input

class MACUNet(nn.Module):
    def __init__(self, LayerNo, num_feature):
        super(MACUNet, self).__init__()
        self.LayerNo = LayerNo
        onelayer = []
        n_feat = 32 - 1

        for i in range(LayerNo):
            num_feature = 32 + i * 32 # CMEM channel features
            onelayer.append(BasicBlock(num_feature))

        self.fcs = nn.ModuleList(onelayer)
        self.fe = nn.Conv2d(1, n_feat, 3, padding=1, bias=True)


    def forward(self, x0, sinogram, theta):
        # initialize the result
        x = x0
        z = self.fe(x) # SIMIB

        prev_features = torch.zeros(x.shape[0], 0, x.shape[2], x.shape[3]).to(x.device)

        for i in range(self.LayerNo):
            x_dual ,x_features = self.fcs[i](x, z, prev_features,theta, sinogram)
            x = x_dual[:, :1, :, :] # 1 feature
            z = x_dual[:, 1:, :, :] # SI: c-1 features

            prev_features = torch.cat([prev_features, x_features], 1)


        xnew = x
        return xnew
