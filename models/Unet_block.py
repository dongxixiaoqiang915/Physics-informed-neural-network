import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import initialization
from copy import deepcopy

# sub-parts of the pruning U-net++ (Dense_unet) model

def lrelu():
    return nn.LeakyReLU(1e-2, inplace=True)


def relu():
    return nn.ReLU(inplace=True)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, ks, drop_rate = 0, nl = 'relu'):
        super(double_conv, self).__init__()
        nll = relu if nl == 'relu' else lrelu
        layers = [
            nn.Conv2d(in_ch, out_ch, ks, padding=1),
            nn.BatchNorm2d(out_ch),
            nll()]
        
        if drop_rate>0:
            layers += [nn.Dropout(drop_rate)]
        self.features = nn.Sequential(*layers)        

    def forward(self, x):
        x = self.features(x)
        return x

class double_conv_last(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, ks, drop_rate = 0, nl = 'lrelu'):
        super(double_conv_last, self).__init__()
        nll = relu if nl == 'relu' else lrelu
        layers = [
            nn.Conv2d(in_ch, out_ch, ks, padding=1),
            nn.BatchNorm2d(out_ch),
            nll()]

        
        if drop_rate>0:
            layers += [nn.Dropout(drop_rate)]
        self.features = nn.Sequential(*layers)        

    def forward(self, x):
        x = self.features(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, ks):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, ks)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, ks, drop_rate = 0, conv_op = nn.Conv2d,
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None, num_pool=2,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False):
        super(down, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        
        #maxpooling for 2D and 3D cases
        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, ks)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    """
    up sampling layer
    """
    def __init__(self, in_ch, out_ch, ks, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, ks)
        self.mpconv = nn.Sequential(
            double_conv(in_ch, out_ch, ks)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.mpconv(x)
        return x

class up_dense(nn.Module):
    """
    up sampling layer of denseblock
    """
    def __init__(self, in_ch, out_ch, ks, bilinear=True):
        super(up_dense, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, ks)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        # x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        ks_out = 1
        pad_out = 0
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size = ks_out,padding = pad_out)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class up_last(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_last, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv_last(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class basic_Convblock(nn.Module):
    """
    basic block used in denselayer
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(basic_Convblock, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Dense_Block(nn.Module):
    def __init__(self, nf, growth_rate, 
                 norm_op=nn.BatchNorm2d,norm_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 conv_op=nn.Conv2d, conv_kwargs=None, basic_block=basic_Convblock, up_flag=False):
        super(Dense_Block, self).__init__()

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        
        self.growth_rate = growth_rate
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = None
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.up_flag = up_flag

        if growth_rate == 3:
            num_features = nf
        elif growth_rate == 2:
            num_features = nf + nf * (growth_rate-1)
        
        self.relu = self.nonlin(**self.nonlin_kwargs)
        self.bn = self.norm_op(nf, **self.norm_op_kwargs)
        self.up1= up_dense(nf*3, nf, ks=3)
        self.up2 = up_dense(nf*3, nf, ks=3)
        self.up3 = up_dense(nf*6, nf*2, ks=3)
        
        if self.up_flag:
            self.blocks = nn.Sequential(
                *([basic_block(num_features, num_features, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, 
                            self.nonlin, self.nonlin_kwargs)] +
                [basic_block(num_features*(_+4), num_features, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs) for _ in range(growth_rate - 1)]))
        else:
            self.blocks = nn.Sequential(
                *([basic_block(num_features, num_features, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, 
                            self.nonlin, self.nonlin_kwargs)] +
                [basic_block(num_features*(_+2), num_features, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs) for _ in range(growth_rate - 1)]))
        
        # self.conv1 = self.conv_op(num_features, num_features, **self.conv_kwargs)
        # self.conv2 = self.conv_op(num_features * 2, num_features, **self.conv_kwargs)
        # self.conv3 = self.conv_op(num_features * 3, num_features, **self.conv_kwargs)

    def forward(self, x, x_down1, x_down2=None):
    # def forward(self, x):  #clear
        # if self.growth_rate == 3:
        #     x11 = self.relu(self.blocks[0](x))
        #     # Concatenate in channel dimension
        #     c1_dense = self.relu(torch.cat([x, x11], 1))
        #     x12 = self.relu(self.blocks[1](c1_dense))
        #     c2_dense = self.relu(torch.cat([x, x11, x12], 1))
        #     x = self.relu(self.blocks[2](c2_dense))

        # elif self.growth_rate == 2:
        #     x21 = self.relu(self.blocks[0](x))
        #     # Concatenate in channel dimension
        #     c1_dense = self.relu(torch.cat([x, x21], 1))
        #     x = self.relu(self.blocks[1](c1_dense))

        # return x

        if self.growth_rate == 3:
            x11 = self.relu(self.blocks[0](x))
            # Concatenate in channel dimension
            if self.up_flag:
                x12 = self.up1(x_down1, x11)
            else:
                x12 = x11
            c1_dense = self.relu(torch.cat([x, x12], 1))
            x12 = self.relu(self.blocks[1](c1_dense))
            if self.up_flag:
                x13 = self.up2(x_down2, x12)
            else:
                x13 = x12
            c2_dense = self.relu(torch.cat([x, x11, x13], 1))
            x = self.relu(self.blocks[2](c2_dense))
            return x

        elif self.growth_rate == 2:
            x21 = self.relu(self.blocks[0](x))
            # Concatenate in channel dimension
            if self.up_flag:
                x22 = self.up3(x_down1, x21)
            else:
                x22 = x21
            c1_dense = self.relu(torch.cat([x, x22], 1))
            x = self.relu(self.blocks[1](c1_dense))
            return x21, x

        

    #upsample layer in densenet layer
    class denseupsample(nn.Module):
        def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):

            super(denseupsample, self).__init__()
            self.align_corners = align_corners
            self.mode = mode
            self.scale_factor = scale_factor
            self.size = size

        def forward(self, x):
            return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                            align_corners=self.align_corners)

    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp


    class BasicResidualBlock(nn.Module):
        def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
            """
            This is the conv bn nonlin conv bn nonlin kind of block
            :param in_planes:
            :param out_planes:
            :param props:
            :param override_stride:
            """
            super().__init__()

            self.kernel_size = kernel_size
            props['conv_op_kwargs']['stride'] = 1

            self.stride = stride
            self.props = props
            self.out_planes = out_planes
            self.in_planes = in_planes

            if stride is not None:
                kwargs_conv1 = deepcopy(props['conv_op_kwargs'])
                kwargs_conv1['stride'] = stride
            else:
                kwargs_conv1 = props['conv_op_kwargs']

            self.conv1 = props['conv_op'](in_planes, out_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                                        **kwargs_conv1)
            self.norm1 = props['norm_op'](out_planes, **props['norm_op_kwargs'])
            self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])

            if props['dropout_op_kwargs']['p'] != 0:
                self.dropout = props['dropout_op'](**props['dropout_op_kwargs'])
            else:
                self.dropout = Identity()

            self.conv2 = props['conv_op'](out_planes, out_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                                        **props['conv_op_kwargs'])
            self.norm2 = props['norm_op'](out_planes, **props['norm_op_kwargs'])
            self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])

            if (self.stride is not None and any((i != 1 for i in self.stride))) or (in_planes != out_planes):
                stride_here = stride if stride is not None else 1
                self.downsample_skip = nn.Sequential(props['conv_op'](in_planes, out_planes, 1, stride_here, bias=False),
                                                    props['norm_op'](out_planes, **props['norm_op_kwargs']))
            else:
                self.downsample_skip = lambda x: x

        def forward(self, x):
            residual = x

            out = self.dropout(self.conv1(x))
            out = self.nonlin1(self.norm1(out))

            out = self.norm2(self.conv2(out))

            residual = self.downsample_skip(residual)

            out += residual

            return self.nonlin2(out)


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        """
        This is the conv bn nonlin conv bn nonlin kind of block
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """
        super().__init__()

        if props['dropout_op_kwargs'] is None and props['dropout_op_kwargs'] > 0:
            raise NotImplementedError("ResidualBottleneckBlock does not yet support dropout!")

        self.kernel_size = kernel_size
        props['conv_op_kwargs']['stride'] = 1

        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.bottleneck_planes = out_planes // 4

        if stride is not None:
            kwargs_conv1 = deepcopy(props['conv_op_kwargs'])
            kwargs_conv1['stride'] = stride
        else:
            kwargs_conv1 = props['conv_op_kwargs']

        self.conv1 = props['conv_op'](in_planes, self.bottleneck_planes, [1 for _ in kernel_size], padding=[0 for i in kernel_size],
                                      **kwargs_conv1)
        self.norm1 = props['norm_op'](self.bottleneck_planes, **props['norm_op_kwargs'])
        self.nonlin1 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv2 = props['conv_op'](self.bottleneck_planes, self.bottleneck_planes, kernel_size, padding=[(i - 1) // 2 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm2 = props['norm_op'](self.bottleneck_planes, **props['norm_op_kwargs'])
        self.nonlin2 = props['nonlin'](**props['nonlin_kwargs'])

        self.conv3 = props['conv_op'](self.bottleneck_planes, out_planes, [1 for _ in kernel_size], padding=[0 for i in kernel_size],
                                      **props['conv_op_kwargs'])
        self.norm3 = props['norm_op'](out_planes, **props['norm_op_kwargs'])
        self.nonlin3 = props['nonlin'](**props['nonlin_kwargs'])

        if (self.stride is not None and any((i != 1 for i in self.stride))) or (in_planes != out_planes):
            stride_here = stride if stride is not None else 1
            self.downsample_skip = nn.Sequential(props['conv_op'](in_planes, out_planes, 1, stride_here, bias=False),
                                                 props['norm_op'](out_planes, **props['norm_op_kwargs']))
        else:
            self.downsample_skip = lambda x: x

    def forward(self, x):
        residual = x

        out = self.nonlin1(self.norm1(self.conv1(x)))
        out = self.nonlin2(self.norm2(self.conv2(out)))

        out = self.norm3(self.conv3(out))

        residual = self.downsample_skip(residual)

        out += residual

        return self.nonlin3(out)

    # class Generic_UNetPlusPlus(SegmentationNetwork):
    #     DEFAULT_BATCH_SIZE_3D = 2
    #     DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    #     SPACING_FACTOR_BETWEEN_STAGES = 2
    #     BASE_NUM_FEATURES_3D = 30
    #     MAX_NUMPOOL_3D = 999
    #     MAX_NUM_FILTERS_3D = 320

    #     DEFAULT_PATCH_SIZE_2D = (256, 256)
    #     BASE_NUM_FEATURES_2D = 30
    #     DEFAULT_BATCH_SIZE_2D = 50
    #     MAX_NUMPOOL_2D = 999
    #     MAX_FILTERS_2D = 480

    #     use_this_for_batch_size_computation_2D = 19739648
    #     use_this_for_batch_size_computation_3D = 520000000 * 2  # 505789440

    #     def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
    #                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
    #                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
    #                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
    #                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
    #                  pool_op_kernel_sizes=None, conv_kernel_sizes=None,
    #                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
    #                 max_num_features=None, basic_block=basic_Convblock,
    #                 seg_output_use_bias=False):
    #         """
    #         basically more flexible than v1, architecture is the same
    #         Does this look complicated? Nah bro. Functionality > usability
    #         This does everything you need, including world peace.
    #         Questions? -> f.isensee@dkfz.de
    #         """
    #         super(Generic_UNetPlusPlus, self).__init__()
    #         self.convolutional_upsampling = convolutional_upsampling
    #         self.convolutional_pooling = convolutional_pooling
    #         self.upscale_logits = upscale_logits
    #         if nonlin_kwargs is None:
    #             nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
    #         if dropout_op_kwargs is None:
    #             dropout_op_kwargs = {'p': 0.5, 'inplace': True}
    #         if norm_op_kwargs is None:
    #             norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

    #         self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

    #         self.nonlin = nonlin
    #         self.nonlin_kwargs = nonlin_kwargs
    #         self.dropout_op_kwargs = dropout_op_kwargs
    #         self.norm_op_kwargs = norm_op_kwargs
    #         self.conv_op = conv_op
    #         self.norm_op = norm_op
    #         self.dropout_op = dropout_op
    #         self.num_classes = num_classes
    #         self._deep_supervision = deep_supervision
    #         self.do_ds = deep_supervision

    #         if conv_op == nn.Conv2d:
    #             upsample_mode = 'bilinear'
    #             pool_op = nn.MaxPool2d
    #             transpconv = nn.ConvTranspose2d
    #             if pool_op_kernel_sizes is None:
    #                 pool_op_kernel_sizes = [(2, 2)] * num_pool
    #             if conv_kernel_sizes is None:
    #                 conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
    #         elif conv_op == nn.Conv3d:
    #             upsample_mode = 'trilinear'
    #             pool_op = nn.MaxPool3d
    #             transpconv = nn.ConvTranspose3d
    #             if pool_op_kernel_sizes is None:
    #                 pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
    #             if conv_kernel_sizes is None:
    #                 conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
    #         else:
    #             raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

    #         self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
    #         self.pool_op_kernel_sizes = pool_op_kernel_sizes
    #         self.conv_kernel_sizes = conv_kernel_sizes

    #         self.conv_pad_sizes = []
    #         for krnl in self.conv_kernel_sizes:
    #             self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

    #         if max_num_features is None:
    #             if self.conv_op == nn.Conv3d:
    #                 self.max_num_features = self.MAX_NUM_FILTERS_3D
    #             else:
    #                 self.max_num_features = self.MAX_FILTERS_2D
    #         else:
    #             self.max_num_features = max_num_features

    #         self.conv_blocks_context = []
    #         # self.conv_blocks_localization = []
    #         self.loc0 = []
    #         self.loc1 = []
    #         self.loc2 = []
    #         self.loc3 = []
    #         self.loc4 = []
    #         self.td = []
    #         self.up0 = []
    #         self.up1 = []
    #         self.up2 = []
    #         self.up3 = []
    #         self.up4 = []
    #         # self.tu = []
    #         self.seg_outputs = []

    #         output_features = base_num_features
    #         input_features = input_channels

    #         for d in range(num_pool):
    #             # determine the first stride
    #             if d != 0 and self.convolutional_pooling:
    #                 first_stride = pool_op_kernel_sizes[d - 1]
    #             else:
    #                 first_stride = None

    #             self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
    #             self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
    #             # add convolutions
    #             self.conv_blocks_context.append(basic_Convblock(input_features, output_features, num_conv_per_stage,
    #                                                             self.conv_op, self.conv_kwargs, self.norm_op,
    #                                                             self.norm_op_kwargs, self.dropout_op,
    #                                                             self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
    #                                                             first_stride, basic_block=basic_block))
    #             if not self.convolutional_pooling:
    #                 self.td.append(pool_op(pool_op_kernel_sizes[d]))
    #             input_features = output_features
    #             output_features = int(np.round(output_features * feat_map_mul_on_downscale))

    #             output_features = min(output_features, self.max_num_features)

    #         # now the bottleneck.
    #         # determine the first stride
    #         if self.convolutional_pooling:
    #             first_stride = pool_op_kernel_sizes[-1]
    #         else:
    #             first_stride = None

    #         # the output of the last conv must match the number of features from the skip connection if we are not using
    #         # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
    #         # done by the transposed conv
    #         if self.convolutional_upsampling:
    #             final_num_features = output_features
    #         else:
    #             final_num_features = self.conv_blocks_context[-1].output_channels

    #         self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
    #         self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
    #         self.conv_blocks_context.append(nn.Sequential(
    #             basic_Convblock(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
    #                             self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
    #                             self.nonlin_kwargs, first_stride, basic_block=basic_block),
    #             basic_Convblock(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
    #                             self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
    #                             self.nonlin_kwargs, basic_block=basic_block)))

    #         # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
    #         if not dropout_in_localization:
    #             old_dropout_p = self.dropout_op_kwargs['p']
    #             self.dropout_op_kwargs['p'] = 0.0

    #         # now lets build the localization pathway
    #         encoder_features = final_num_features
    #         self.loc0, self.up0, encoder_features = self.create_nest(0, num_pool, final_num_features, num_conv_per_stage,
    #                                                                 basic_block, transpconv)
    #         self.loc1, self.up1, encoder_features1 = self.create_nest(1, num_pool, encoder_features, num_conv_per_stage,
    #                                                                 basic_block, transpconv)
    #         self.loc2, self.up2, encoder_features2 = self.create_nest(2, num_pool, encoder_features1, num_conv_per_stage,
    #                                                                 basic_block, transpconv)
    #         self.loc3, self.up3, encoder_features3 = self.create_nest(3, num_pool, encoder_features2, num_conv_per_stage,
    #                                                                 basic_block, transpconv)
    #         self.loc4, self.up4, encoder_features4 = self.create_nest(4, num_pool, encoder_features3, num_conv_per_stage,
    #                                                                 basic_block, transpconv)

    #         self.seg_outputs.append(conv_op(self.loc0[-1][-1].output_channels, num_classes,
    #                                         1, 1, 0, 1, 1, seg_output_use_bias))
    #         self.seg_outputs.append(conv_op(self.loc1[-1][-1].output_channels, num_classes,
    #                                         1, 1, 0, 1, 1, seg_output_use_bias))
    #         self.seg_outputs.append(conv_op(self.loc2[-1][-1].output_channels, num_classes,
    #                                         1, 1, 0, 1, 1, seg_output_use_bias))
    #         self.seg_outputs.append(conv_op(self.loc3[-1][-1].output_channels, num_classes,
    #                                         1, 1, 0, 1, 1, seg_output_use_bias))
    #         self.seg_outputs.append(conv_op(self.loc4[-1][-1].output_channels, num_classes,
    #                                         1, 1, 0, 1, 1, seg_output_use_bias))

    #         self.upscale_logits_ops = []
    #         cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
    #         for usl in range(num_pool):
    #             if self.upscale_logits:
    #                 self.upscale_logits_ops.append(up(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
    #                                                         mode=upsample_mode))
    #             else:
    #                 self.upscale_logits_ops.append(lambda x: x)

    #         if not dropout_in_localization:
    #             self.dropout_op_kwargs['p'] = old_dropout_p

    #         # register all modules properly
    #         # self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
    #         self.loc0 = nn.ModuleList(self.loc0)
    #         self.loc1 = nn.ModuleList(self.loc1)
    #         self.loc2 = nn.ModuleList(self.loc2)
    #         self.loc3 = nn.ModuleList(self.loc3)
    #         self.loc4 = nn.ModuleList(self.loc4)
    #         self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
    #         self.td = nn.ModuleList(self.td)
    #         self.up0 = nn.ModuleList(self.up0)
    #         self.up1 = nn.ModuleList(self.up1)
    #         self.up2 = nn.ModuleList(self.up2)
    #         self.up3 = nn.ModuleList(self.up3)
    #         self.up4 = nn.ModuleList(self.up4)
    #         self.seg_outputs = nn.ModuleList(self.seg_outputs)
    #         if self.upscale_logits:
    #             self.upscale_logits_ops = nn.ModuleList(
    #                 self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

    #         if self.weightInitializer is not None:
    #             self.apply(self.weightInitializer)
    #             # self.apply(print_module_training_status)

    #     def forward(self, x):
    #         # skips = []
    #         seg_outputs = []
    #         x0_0 = self.conv_blocks_context[0](x)
    #         x1_0 = self.conv_blocks_context[1](x0_0)
    #         x0_1 = self.loc4[0](torch.cat([x0_0, self.up4[0](x1_0)], 1))
    #         seg_outputs.append(self.final_nonlin(self.seg_outputs[-1](x0_1)))

    #         x2_0 = self.conv_blocks_context[2](x1_0)
    #         x1_1 = self.loc3[0](torch.cat([x1_0, self.up3[0](x2_0)], 1))
    #         x0_2 = self.loc3[1](torch.cat([x0_0, x0_1, self.up3[1](x1_1)], 1))
    #         seg_outputs.append(self.final_nonlin(self.seg_outputs[-2](x0_2)))

    #         x3_0 = self.conv_blocks_context[3](x2_0)
    #         x2_1 = self.loc2[0](torch.cat([x2_0, self.up2[0](x3_0)], 1))
    #         x1_2 = self.loc2[1](torch.cat([x1_0, x1_1, self.up2[1](x2_1)], 1))
    #         x0_3 = self.loc2[2](torch.cat([x0_0, x0_1, x0_2, self.up2[2](x1_2)], 1))
    #         seg_outputs.append(self.final_nonlin(self.seg_outputs[-3](x0_3)))

    #         x4_0 = self.conv_blocks_context[4](x3_0)
    #         x3_1 = self.loc1[0](torch.cat([x3_0, self.up1[0](x4_0)], 1))
    #         x2_2 = self.loc1[1](torch.cat([x2_0, x2_1, self.up1[1](x3_1)], 1))
    #         x1_3 = self.loc1[2](torch.cat([x1_0, x1_1, x1_2, self.up1[2](x2_2)], 1))
    #         x0_4 = self.loc1[3](torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1[3](x1_3)], 1))
    #         seg_outputs.append(self.final_nonlin(self.seg_outputs[-4](x0_4)))

    #         x5_0 = self.conv_blocks_context[5](x4_0)
    #         x4_1 = self.loc0[0](torch.cat([x4_0, self.up0[0](x5_0)], 1))
    #         x3_2 = self.loc0[1](torch.cat([x3_0, x3_1, self.up0[1](x4_1)], 1))
    #         x2_3 = self.loc0[2](torch.cat([x2_0, x2_1, x2_2, self.up0[2](x3_2)], 1))
    #         x1_4 = self.loc0[3](torch.cat([x1_0, x1_1, x1_2, x1_3, self.up0[3](x2_3)], 1))
    #         x0_5 = self.loc0[4](torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up0[4](x1_4)], 1))
    #         seg_outputs.append(self.final_nonlin(self.seg_outputs[-5](x0_5)))

    #         if self._deep_supervision and self.do_ds:
    #             return tuple([seg_outputs[-1]] + [i(j) for i, j in
    #                                             zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
    #         else:
    #             return seg_outputs[-1]