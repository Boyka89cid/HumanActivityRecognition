from torch import nn
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from fastai.imports import *
from torch.utils.data import DataLoader
from fastai.learner import *
from fastai.metrics import accuracy, accuracy_thresh
from datasets_loadings import TwoStreamTSNDataset, ModelData
from running_parameters import parser
from twostream_models import TwoStreamNetwork
from transformations import Stack, GroupScale, ToTorchFormatTensor, GroupNormalize
import gc

best_precision1 = 0
is_cuda = torch.cuda.is_available() 


def main():
    global args, best_precision1
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        args.num_class = 101
        args.from_videos = False
        args.multi_label = False
        args.basic = True
        init_weights = torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'smart-vision':
        args.num_class = 5
        args.from_videos = True
        args.multi_label = False
        args.basic = True
        init_weights = torch.FloatTensor([0.5, 1.0, 1.0, 0.75, 1.0])
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    # print(args)
    args.modality = ['RGB', 'RGBDiff']
    model = TwoStreamNetwork(args.num_class, args.num_segments, args.modality,
                             base_model=args.arch, freeze_base=args.freeze_base, multi_label=args.multi_label,
                             consensus_type=args.consensus_type, dropout=args.dropout,
                             partial_bn=not args.no_partial_bn)

    crop_size = model.crop_size
    scale_size = crop_size
    policies = model.policies

    first_stream_mean = model.stream_one_mean
    first_stream_std = model.stream_one_std
    first_stream_augmentation = model.stream_one_augmentation

    second_stream_mean = model.stream_two_mean
    second_stream_std = model.stream_two_std
    second_stream_augmentation = model.stream_two_augmentation

    model = torch.nn.DataParallel(model, device_ids=list(range(args.gpus))).cuda() if is_cuda else model
    cudnn.benchmark = True

    if args.modality[1] == 'RGB':
        data_length = 1
    elif args.modality[1] in ['Flow', 'RGBDiff']:
        data_length = 5

    train_dataset = TwoStreamTSNDataset("", args.train_list, num_segments=args.num_segments,
                                        new_length=data_length,
                                        modality=args.modality[1],
                                        image_tmpl=".{:04d}.jpg",
                                        split=1,
                                        basic=args.basic,
                                        arguments=args,
                                        transform=[torchvision.transforms.Compose([
                                            GroupScale((scale_size, scale_size)),
                                            first_stream_augmentation,
                                            Stack(roll=args.arch == 'BNInception'),
                                            ToTorchFormatTensor(div=args.arch != 'BNInception'),
                                            GroupNormalize(first_stream_mean, first_stream_std),
                                        ]), torchvision.transforms.Compose([
                                            GroupScale((scale_size, scale_size)),
                                            second_stream_augmentation,
                                            Stack(roll=args.arch == 'BNInception'),
                                            ToTorchFormatTensor(div=args.arch != 'BNInception'),
                                            GroupNormalize(second_stream_mean, second_stream_std),
                                        ])])

    validation_dataset = TwoStreamTSNDataset("", args.val_list, num_segments=args.num_segments,
                                             new_length=data_length,
                                             modality=args.modality[1],
                                             image_tmpl=".{:04d}.jpg",
                                             split=0,
                                             basic=args.basic,
                                             arguments=args,
                                             random_shift=False,
                                             transform=[torchvision.transforms.Compose([
                                                 GroupScale((scale_size, scale_size)),
                                                 Stack(roll=args.arch == 'BNInception'),
                                                 ToTorchFormatTensor(div=args.arch != 'BNInception'),
                                                 GroupNormalize(first_stream_mean, first_stream_std),
                                             ]), torchvision.transforms.Compose([
                                                 GroupScale((scale_size, scale_size)),
                                                 Stack(roll=args.arch == 'BNInception'),
                                                 ToTorchFormatTensor(div=args.arch != 'BNInception'),
                                                 GroupNormalize(second_stream_mean, second_stream_std),
                                             ])])

    training_loader = DataLoader(train_dataset,
                                 batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.workers, pin_memory=False)

    validation_loader = DataLoader(validation_dataset,
                                   batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=False)
    # Initialize fastai methods with loaders and model
    md = ModelData(args.model_path, training_loader, validation_loader)
    learn = Learner.from_model_data(model, md, clip=args.clip_gradient)
    # print(learn.models.model)

    del training_loader
    del validation_loader
    del md

    gc.collect()

    if args.multi_label:
        learn.crit = nn.BCELoss().cuda() if is_cuda else nn.BCELoss()
    else:
        learn.crit = nn.CrossEntropyLoss(weight=init_weights).cuda() if is_cuda else nn.CrossEntropyLoss(
            weight=init_weights)

    if args.resume or args.transfer_learning:
        if os.path.isfile(args.resume):
            if not args.transfer_learning:
                learn.load(args.resume[:-3])
            else:
                sd = torch.load(learn.get_model_path(args.resume[:-3]), map_location=lambda storage, loc: storage)
                sd = {k: v for k, v in sd.items() if 'classifier' not in k and 'fc' not in k}
                learn.model.load_state_dict(sd, strict=False)
                del sd

        else:
            print(("=> Path not found at '{}'".format(args.resume)))
    learn.metrics = [accuracy_thresh(0.5)] if args.multi_label else [accuracy]
    learn.fit(args.lr, args.epochs,
              wds=args.weight_decay,
              cycle_len=args.cycle_length,
              cycle_mult=args.cycle_mult,
              best_save_name='best')


if __name__ == '__main__':
    main()

"""
Example: To run this experiment run the lines below.

First Run:
CUDA_VISIBLE_DEVICES=0 \
python  two_stream.py \
--dataset smart-vision \
--data_path /datasets/charades/data/videos/Charades_v1_480 \
--model_path /datasets/smart-vision/models/charades/classifier-a/tsn_mobilenet-v2/experiment \
--train_list /datasets/charades/meta-data/charades_classifier-a_stable_balanced_v2.json \
--val_list /datasets/charades/meta-data/charades_classifier-a_validation.json \
--arch mobilenet_v2 --num_segments 3 --no_partial_bn --dropout 0.8 \
--lr 1e-2 --epochs 5 --cycle_length 1 --cycle_mult 2 --weight_decay 5e-4 --clip_gradient 20 \
--gpus 1 -b 20 -j 56 \
"""
