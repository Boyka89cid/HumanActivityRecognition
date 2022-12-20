import torch
from torch import nn
import torchvision
from torch.nn.init import normal, constant
from consensus import ConsensusPerModule
from transformations import GroupRandomHorizontalFlip, Translate, Skew
from transformations import Brightness, Contrast, Saturation, Hue, Noise
from transformations import Shear
from mobilenet_v2 import mobilenet_v2
from squeezenet import squeezenet1_1
import numpy as np


class TwoStreamNetwork(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101',
                 consensus_type='avg',
                 multi_label=False,
                 dropout=0.8,
                 freeze_base=True,
                 partial_bn=True):
        super(TwoStreamNetwork, self).__init__()
        self.multi_label = multi_label
        self.consensus_type = consensus_type

        self.stream_one_model = TemporalSegmentNetworks(num_class, num_segments, modality[0],
                                                        base_model=base_model,
                                                        consensus_type=consensus_type,
                                                        dropout=dropout,
                                                        freeze_base=freeze_base,
                                                        partial_bn=partial_bn)
        self.stream_two_model = TemporalSegmentNetworks(num_class, num_segments, modality[1],
                                                        base_model=base_model,
                                                        consensus_type=consensus_type,
                                                        dropout=dropout,
                                                        freeze_base=freeze_base,
                                                        partial_bn=partial_bn)

        self.consensus = ConsensusPerModule(consensus_type)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.enable_partial_bn(True)

        self.crop_size = self.stream_one_model.crop_size
        self.scale_size = self.stream_one_model.scale_size
        self.policies = self.stream_one_model.get_optim_policies()

        self.stream_one_mean = self.stream_one_model.input_mean
        self.stream_one_std = self.stream_one_model.input_std
        self.stream_one_augmentation = self.stream_one_model.get_augmentation()

        self.stream_two_mean = self.stream_two_model.input_mean
        self.stream_two_std = self.stream_two_model.input_std
        self.stream_two_augmentation = self.stream_two_model.get_augmentation()

    def enable_partial_bn(self, enable):
        self._enable_pbn = enable

    def forward(self, stream_one, stream_two):
        stream_one_output = self.stream_one_model(stream_one)
        stream_two_output = self.stream_two_model(stream_two)
        merged_output = torch.cat((stream_one_output, stream_two_output), 1)

        # print(stream_one_output.shape, stream_two_output.shape, merged_output.shape)
        output = self.consensus(merged_output).squeeze(1)
        # output = self.loss_fn(output)

        return output


class TemporalSegmentNetworks(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 freeze_base=True,
                 heavy_aug=False,
                 crop_num=1, partial_bn=True):
        super(TemporalSegmentNetworks, self).__init__()
        self.base_model_name = base_model
        self.modality = modality
        self.num_segments = num_segments
        self.num_class = num_class
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.freeze_base = freeze_base
        self.heavy_aug = heavy_aug

        self.modified_models = ['mobilenet_v2', 'squeezenet1_1']

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TemporalSegmentNetworks with base model: {}.
TemporalSegmentNetworks Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(self.base_model_name, self.modality, self.num_segments, self.new_length, consensus_type,
                   self.dropout)))

        self._prepare_base_model(self.base_model_name)

        if self.base_model_name not in self.modified_models:
            self._prepare_tsn()

        if self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        # self.consensus = ConsensusPerModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.enable_partial_bn(True)

    '''
    Below Function has been referred from
    https://github.com/yjxiong/tsn-pytorch/blob/master/models.py
    '''

    def _prepare_tsn(self):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, self.num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, self.num_class)

        std = 0.001
        if self.new_fc is None:
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)

    '''
    Below Function has been referred from
    https://github.com/yjxiong/tsn-pytorch/blob/master/models.py
    '''

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std += [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif 'mobilenet_v2' in base_model:
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model = mobilenet_v2(pretrained=True,
                                           n_class=self.num_class,
                                           input_size=self.input_size,
                                           width_mult=1.0,
                                           dropout_ratio=self.dropout)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std += [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif 'squeezenet' in base_model:

            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std += [np.mean(self.input_std) * 2] * 3 * self.new_length

            self.base_model = squeezenet1_1(pretrained=True, num_classes=self.num_class, dropout=self.dropout)

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True, train_classification_layers=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TemporalSegmentNetworks, self).train(mode)
        count = 0

        # pdb.set_trace()
        if self.freeze_base:
            # print("Freezing all base_model weights except classification layers")
            for m in self.base_model.modules():
                # freeze all layers except classification layers
                if train_classification_layers:
                    if isinstance(m, nn.BatchNorm2d):
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
                    if isinstance(m, nn.Conv2d):
                        m.weight.requires_grad = False
        else:
            for param in self.base_model.parameters():
                param.requires_grad = True
            if self._enable_pbn:
                # print("Freezing BatchNorm2D except the first one.")
                for m in self.base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= (2 if self._enable_pbn else 1):
                            m.eval()

                            # shutdown update in frozen mode
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
                            # for i, p in enumerate(reversed(list(self.base_model.parameters()))):
                            #    p.requires_grad = True if i < 23 else False

    def enable_partial_bn(self, enable):
        self._enable_pbn = enable

    '''
    Below Function has been referred from
    https://github.com/yjxiong/tsn-pytorch/blob/master/models.py
    '''
    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, _input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            _input = self._get_diff(_input)

        base_out = self.base_model(_input.view((-1, sample_len) + _input.size()[-2:]))

        if self.dropout > 0 and self.base_model_name not in self.modified_models:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        # output = self.consensus(base_out)
        # return output.squeeze(1)
        return base_out

    '''
    Below Function has been referred from
    https://github.com/yjxiong/tsn-pytorch/blob/master/models.py
    '''
    def _get_diff(self, _input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = _input.view((-1, self.num_segments, self.new_length + 1, input_c,) + tuple(_input.size()[2:]))
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    '''
    Below Function has been referred from
    https://github.com/yjxiong/tsn-pytorch/blob/master/models.py
    '''
    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        # print(list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0])
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat(
                (params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            if not self.heavy_aug:
                return torchvision.transforms.Compose([  # GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_flow=False),
                    Translate(probability=0.5, translate=(0.05, 0.05)),
                    Skew(probability=0.5, magnitude=0.25),
                    Shear(probability=0.25, max_shear_left=8, max_shear_right=8),
                    Brightness(probability=0.5, brightness=0.5),
                    Contrast(probability=0.5, contrast=0.5),
                    Saturation(probability=0.25, saturation=0.5),
                    Hue(probability=0.25, hue=0.1),
                    Noise(probability=.25),
                ])
            else:
                return torchvision.transforms.Compose([  # GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                    GroupRandomHorizontalFlip(is_flow=False),
                    Translate(probability=0.5, translate=(0.1, 0.1)),
                    Skew(probability=0.5, magnitude=0.25),
                    Shear(probability=0.5, max_shear_left=16, max_shear_right=16),
                    Brightness(probability=1, brightness=1.0),
                    Contrast(probability=1, contrast=1.0),
                    Saturation(probability=0.5, saturation=0.5),
                    Hue(probability=0.5, hue=0.1),
                    Noise(probability=.5),
                ])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([  # GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                GroupRandomHorizontalFlip(is_flow=False),
                Translate(probability=0.5, translate=(0.05, 0.05)),
                Skew(probability=0.5, magnitude=0.25),
                Shear(probability=0.25, max_shear_left=8, max_shear_right=8),
            ])
