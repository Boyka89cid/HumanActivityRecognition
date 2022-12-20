import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from fastai.imports import *
from fastai.conv_learner import ConvLearner
from fastai.learner import Learner
from fastai.metrics import accuracy, accuracy_thresh
from fastai.dataset import ImageData, ModelData
from datasets_loadings import TwoStreamTSNDataset
from running_parameters import parser
from twostream_models import TwoStreamNetwork
from transformations import IdentityTransform, \
    GroupNormalize, Stack, GroupCenterCrop, GroupScale, ToTorchFormatTensor
from fastai.core import to_np, VV
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn
import gc

best_prec1 = 0
use_cuda = torch.cuda.is_available()


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
        class_names = list(range(num_class))
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'smart-vision':
        args.num_class = 5
        num_class = args.num_class
        class_names = ['sit', 'stand', 'lie-down', 'walk', 'background']
        args.multi_label = False
        args.from_videos = True
        args.basic = True
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    args.modality = ['RGB', 'RGBDiff']
    model = TwoStreamNetwork(num_class, args.num_segments, args.modality,
                             base_model=args.arch, freeze_base=args.freeze_base, multi_label=args.multi_label,
                             consensus_type=args.consensus_type, dropout=args.dropout,
                             partial_bn=not args.no_partial_bn)

    crop_size = model.crop_size
    scale_size = 224
    policies = model.policies

    stream_one_mean = model.stream_one_mean
    stream_one_std = model.stream_one_std
    stream_one_augmentation = model.stream_one_augmentation

    stream_two_mean = model.stream_two_mean
    stream_two_std = model.stream_two_std
    stream_two_augmentation = model.stream_two_augmentation

    model = torch.nn.DataParallel(model, device_ids=list(range(args.gpus))).cuda() if use_cuda else model
    cudnn.benchmark = True

    if args.modality[1] == 'RGB':
        data_length = 1
    elif args.modality[1] in ['Flow', 'RGBDiff']:
        data_length = 5

    train_ds = TwoStreamTSNDataset("", args.train_list, num_segments=args.num_segments,
                                   new_length=data_length,
                                   modality=args.modality[1],
                                   image_tmpl=".{:04d}.jpg",
                                   split=1,
                                   basic=args.basic,
                                   arguments=args,
                                   transform=[torchvision.transforms.Compose([
                                       GroupScale((scale_size, scale_size)),
                                       stream_one_augmentation,
                                       Stack(roll=args.arch == 'BNInception'),
                                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                                       # GroupNormalize(stream_one_mean, stream_one_std),
                                   ]), torchvision.transforms.Compose([
                                       GroupScale((scale_size, scale_size)),
                                       stream_two_augmentation,
                                       Stack(roll=args.arch == 'BNInception'),
                                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                                       # GroupNormalize(stream_two_mean, stream_two_std),
                                   ])])

    val_ds = TwoStreamTSNDataset("", args.val_list, num_segments=args.num_segments,
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
                                     # GroupNormalize(stream_one_mean, stream_one_std),
                                 ]), torchvision.transforms.Compose([
                                     GroupScale((scale_size, scale_size)),
                                     Stack(roll=args.arch == 'BNInception'),
                                     ToTorchFormatTensor(div=args.arch != 'BNInception'),
                                     # GroupNormalize(stream_two_mean, stream_two_std),
                                 ])])

    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=False)

    md = ModelData(args.model_path, train_loader, val_loader)
    learn = Learner.from_model_data(model, md)

    del train_loader
    # del val_loader
    # del md

    gc.collect()

    if args.multi_label:
        learn.crit = nn.BCELoss().cuda() if use_cuda else nn.BCELoss()
    else:
        learn.crit = nn.CrossEntropyLoss().cuda() if use_cuda else nn.CrossEntropyLoss()

    if args.resume:
        if os.path.isfile(args.resume):
            learn.load(args.resume[:-3])
        else:
            print(("=> Path not found at '{}'".format(args.resume)))
    learn.metrics = [accuracy_thresh(0.5)] if args.multi_label else [accuracy]

    x1, x2, y = next(iter(md.val_dl))
    learn.model.eval()
    preds = to_np(F.softmax(learn.model(VV(x1), VV(x2))))
    print(preds, np.argmax(preds), y)
    trans = torchvision.transforms.ToPILImage()
    # pdb.set_trace()
    trans(x2[0, 6:9].cpu()).show()

    y_hat, y_true = learn.predict_with_targs()
    y_hat = np.argmax(y_hat, axis=1)
    print("Accuracy: " + str(accuracy_score(y_true, y_hat)))
    # plot_cm(y_true, y_hat, args.model_path)
    plot_confusion_matrix(y_true, y_hat, class_names, args.model_path, True)

    # for i in range(0,y_hat.shape[0]):
    #     print(y_hat[i], y_true[i])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_cm(labels, predictions, path):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(labels, predictions)
    cm = (cm.astype(float) / cm.astype(float).sum(axis=1)[:, np.newaxis]) * 100
    cm = cm.round(decimals=0).astype(int)
    # plt.rc('axes', labelsize=30)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=30)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=30)  # fontsize of the tick labels
    # plt.rcParams['figure.figsize'] = [60, 60]
    # plt.rcParams['font.size'] = 30
    # plt.yticks(rotation=90)
    # plt.xticks(rotation=0)
    plt.figure()
    sn.heatmap(cm, annot=True, fmt='d', cbar=False, square=True, cmap="YlGnBu")
    plt.savefig(os.path.join(path, 'confusion_matrix.png'), dpi=300)


def plot_confusion_matrix(labels, predictions, classes, path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    cm = confusion_matrix(labels, predictions)
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(os.path.join(path, 'confusion_matrix.png'), dpi=300)
    plt.show()


if __name__ == '__main__':
    main()

"""
Example: To run this experiment run the lines below.

First Run:
python two_stream.py \
--dataset ucf101 --modality RGBDiff \
--data_path /data/ai-bandits/datasets/ucf101_sample/images/rgb \
--model_path /data/ai-bandits/datasets/smart-vision/models \
--train_list /home/avn3r/code/python/tsn-keras/data/ucf101_sample/rgb/train_videos.csv \
--val_list /home/avn3r/code/python/tsn-keras/data/ucf101_sample/rgb/validation_videos.csv \
--arch mobilenet_v2 --num_segments 3 \
--lr 1e-2 --epochs 1 --cycle_length 10 \
--gpus 1 -b 16 -j 6 --dropout 0.8 --freeze_base \




Second Run:
python two_stream.py \
--dataset ucf101 --modality RGBDiff \
--data_path /data/ai-bandits/datasets/ucf101_sample/images/rgb \
--model_path /data/ai-bandits/datasets/smart-vision/models \
--train_list /home/avn3r/code/python/tsn-keras/data/ucf101_sample/rgb/train_videos.csv \
--val_list /home/avn3r/code/python/tsn-keras/data/ucf101_sample/rgb/validation_videos.csv \
--arch mobilenet_v2 --num_segments 3 \
--lr 1e-2 --epochs 1 --cycle_length 20 \
--gpus 1 -b 16 -j 6 --dropout 0.8 --resume \
"""
