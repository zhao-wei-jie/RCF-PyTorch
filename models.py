import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from convnext import convnext_tiny
import sys

class RCF(nn.Module):
    def __init__(self, pretrained=None):
        super(RCF, self).__init__()
        self.conv1_1 = nn.Conv2d(  3,  64, 3, padding=1, dilation=1)
        self.conv1_2 = nn.Conv2d( 64,  64, 3, padding=1, dilation=1)
        self.conv2_1 = nn.Conv2d( 64, 128, 3, padding=1, dilation=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1, dilation=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1, dilation=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        self.act = nn.ReLU(inplace=True)

        self.bn=nn.ModuleDict()#添加bn层,防止梯度爆炸
        bn_layer=[1,2,3,4,5]
        bn_time=[2,2,3,3,3]
        bn_channel=[64,128,256,512,512]
        for l,t,c in zip(bn_layer,bn_time,bn_channel):
            for i in range(t):
                self.bn[str(l)+'-'+str(i+1)]=nn.BatchNorm2d(c)
        # print(self.bn['1-1'])

        self.conv1_1_down = nn.Conv2d( 64, 21, 1)
        self.conv1_2_down = nn.Conv2d( 64, 21, 1)
        self.conv2_1_down = nn.Conv2d(128, 21, 1)
        self.conv2_2_down = nn.Conv2d(128, 21, 1)
        self.conv3_1_down = nn.Conv2d(256, 21, 1)
        self.conv3_2_down = nn.Conv2d(256, 21, 1)
        self.conv3_3_down = nn.Conv2d(256, 21, 1)
        self.conv4_1_down = nn.Conv2d(512, 21, 1)
        self.conv4_2_down = nn.Conv2d(512, 21, 1)
        self.conv4_3_down = nn.Conv2d(512, 21, 1)
        self.conv5_1_down = nn.Conv2d(512, 21, 1)
        self.conv5_2_down = nn.Conv2d(512, 21, 1)
        self.conv5_3_down = nn.Conv2d(512, 21, 1)

        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        self.score_fuse = nn.Conv2d(5, 1, 1)

        self.weight_deconv2 = self._make_bilinear_weights( 4, 1).cuda()
        self.weight_deconv3 = self._make_bilinear_weights( 8, 1).cuda()
        self.weight_deconv4 = self._make_bilinear_weights(16, 1).cuda()
        self.weight_deconv5 = self._make_bilinear_weights(16, 1).cuda()

        # init weights
        self.apply(self._init_weights)
        if pretrained is not None:
            vgg16 = sio.loadmat(pretrained)
            torch_params = self.state_dict()

            for k in vgg16.keys():
                name_par = k.split('-')
                size = len(name_par)
                if size == 2:
                    name_space = name_par[0] + '.' + name_par[1]
                    data = np.squeeze(vgg16[k])
                    torch_params[name_space] = torch.from_numpy(data)
            self.load_state_dict(torch_params)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
            if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
                nn.init.constant_(m.weight, 0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # Based on HED implementation @ https://github.com/xwjabc/hed
    def _make_bilinear_weights(self, size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        w.requires_grad = False
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    w[i, j] = filt
        return w

    # Based on BDCN implementation @ https://github.com/pkuCactus/BDCN
    def _crop(self, data, img_h, img_w, crop_h, crop_w):
        _, _, h, w = data.size()
        assert(img_h <= h and img_w <= w)
        data = data[:, :, crop_h:crop_h + img_h, crop_w:crop_w + img_w]
        return data

    def forward(self, x):
        img_h, img_w = x.shape[2], x.shape[3]

        conv1_1 = self.act(self.bn['1-1'](self.conv1_1(x)))
        conv1_2 = self.act(self.bn['1-2'](self.conv1_2(conv1_1)))
        pool1   = self.pool1(conv1_2)
        conv2_1 = self.act(self.bn['2-1'](self.conv2_1(pool1)))
        conv2_2 = self.act(self.bn['2-2'](self.conv2_2(conv2_1)))
        pool2   = self.pool2(conv2_2)
        conv3_1 = self.act(self.bn['3-1'](self.conv3_1(pool2)))
        conv3_2 = self.act(self.bn['3-2'](self.conv3_2(conv3_1)))
        conv3_3 = self.act(self.bn['3-3'](self.conv3_3(conv3_2)))
        pool3   = self.pool3(conv3_3)
        conv4_1 = self.act(self.bn['4-1'](self.conv4_1(pool3)))
        conv4_2 = self.act(self.bn['4-2'](self.conv4_2(conv4_1)))
        conv4_3 = self.act(self.bn['4-3'](self.conv4_3(conv4_2)))
        pool4   = self.pool4(conv4_3)
        conv5_1 = self.act(self.bn['5-1'](self.conv5_1(pool4)))
        conv5_2 = self.act(self.bn['5-2'](self.conv5_2(conv5_1)))
        conv5_3 = self.act(self.bn['5-3'](self.conv5_3(conv5_2)))

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        out1 = self.score_dsn1(conv1_1_down + conv1_2_down)
        out2 = self.score_dsn2(conv2_1_down + conv2_2_down)
        out3 = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        out4 = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        out5 = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)

        out2 = F.conv_transpose2d(out2, self.weight_deconv2, stride=2)
        out3 = F.conv_transpose2d(out3, self.weight_deconv3, stride=4)
        out4 = F.conv_transpose2d(out4, self.weight_deconv4, stride=8)
        out5 = F.conv_transpose2d(out5, self.weight_deconv5, stride=8)

        out2 = self._crop(out2, img_h, img_w, 1, 1)
        out3 = self._crop(out3, img_h, img_w, 2, 2)
        out4 = self._crop(out4, img_h, img_w, 4, 4)
        out5 = self._crop(out5, img_h, img_w, 0, 0)
        # print(out3.max(),out3.min(),out4.max(),out4.min(),out5.max(),out5.min())
        fuse = torch.cat((out1, out2, out3, out4, out5), dim=1)
        fuse = self.score_fuse(fuse)
        results = [out1, out2, out3, out4, out5, fuse]
        results = [torch.sigmoid(r) for r in results]
        return results

class RCFHead(nn.Module):
    def __init__(self, pretrained=None):
        super(RCFHead, self).__init__()

        self.conv_down=nn.ModuleList()
        self.down_channel=[96,96,192,384,768]
        for i in range(len(self.down_channel)):

            self.conv_down.append(nn.Conv2d( self.down_channel[i], 21, 1))
        
        self.score_dsn=nn.ModuleList()
        for i in range(len(self.down_channel)):
            self.score_dsn.append(nn.Conv2d(21, 1, 1))
        self.score_fuse = nn.Conv2d(len(self.down_channel), 1, 1)

        

        self.weight_deconv2 = self._make_bilinear_weights(8, 1).cuda()
        self.weight_deconv3 = self._make_bilinear_weights(16, 1).cuda()
        self.weight_deconv4 = self._make_bilinear_weights(32, 1).cuda()
        self.weight_deconv5 = self._make_bilinear_weights(48, 1).cuda()

        # init weights
        self.apply(self._init_weights)
        if pretrained is not None:
            vgg16 = sio.loadmat(pretrained)
            torch_params = self.state_dict()

            for k in vgg16.keys():
                name_par = k.split('-')
                size = len(name_par)
                if size == 2:
                    name_space = name_par[0] + '.' + name_par[1]
                    data = np.squeeze(vgg16[k])
                    torch_params[name_space] = torch.from_numpy(data)
            self.load_state_dict(torch_params)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
            if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
                nn.init.constant_(m.weight, 0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # Based on HED implementation @ https://github.com/xwjabc/hed
    def _make_bilinear_weights(self, size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        w.requires_grad = False
        for i in range(num_channels):
            for j in range(num_channels):
                if i == j:
                    w[i, j] = filt
        return w

    # Based on BDCN implementation @ https://github.com/pkuCactus/BDCN
    def _crop(self, data, img_h, img_w, crop_h, crop_w):
        _, _, h, w = data.size()
        assert(img_h <= h and img_w <= w),(img_h,h,img_w,w)
        crop_h=(h-img_h)//2
        crop_w=(w-img_w)//2
        data = data[:, :, crop_h:crop_h + img_h, crop_w:crop_w + img_w]
        return data

    def forward(self, img_h, img_w,out):
        
        assert len(out)==5
        # for o in out:   
        #     print(o.shape)
        out=list(out)
        for i in range(len(self.down_channel)):
            # i=i+1
            out[i]=self.conv_down[i](out[i])#降维至21

        for i in range(len(self.down_channel)):
            # i=i+1
            out[i]=self.score_dsn[i](out[i])#降维至1

        # for o in out:   
        #     print(out[o].shape)        
        
        out[0] = F.conv_transpose2d(out[0], self.weight_deconv2, stride=4)
        out[1] = F.conv_transpose2d(out[1], self.weight_deconv2, stride=4)
        out[2] = F.conv_transpose2d(out[2], self.weight_deconv3, stride=8)
        out[3] = F.conv_transpose2d(out[3], self.weight_deconv4, stride=16)
        out[4] = F.conv_transpose2d(out[4], self.weight_deconv5, stride=32)

        # for o in out:   
        #     print(o.shape)
        
        out[0] = self._crop(out[0], img_h, img_w, 1, 0)
        out[1] = self._crop(out[1], img_h, img_w, 1, 0)
        out[2] = self._crop(out[2], img_h, img_w, 1, 0)
        out[3] = self._crop(out[3], img_h, img_w, 4, 0)
        out[4] = self._crop(out[4], img_h, img_w, 0, 0)
        # for o in out:   
        #     print(o.shape)
        # sys.exit(0)
        # print(out3.max(),out3.min(),out4.max(),out4.min(),out5.max(),out5.min())
        fuse = torch.cat((out[0], out[1], out[2], out[3],out[4]), dim=1)
        fuse = self.score_fuse(fuse)
        results = [out[0], out[1], out[2], out[3],out[4], fuse]
        results = [torch.sigmoid(r) for r in results]
        return results

class NextRCF(nn.Module):
    def __init__(self, pretrained=None):
        super(NextRCF, self).__init__()
        self.backbone=convnext_tiny(pretrained=True)
        self.head=RCFHead()
    def forward(self,x):
        img_h, img_w = x.shape[2], x.shape[3]
        x=self.backbone(x)
        x=self.head(img_h,img_w,x)
        return x

