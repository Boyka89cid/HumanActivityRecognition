import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Two Stream Temporal Segment Networks")

# ======================== Default Values ==========================
CONV_NETS = 'mobilenet_v2'
NUM_VID_SEGMENTS = 3
CONSENSUS_TYPE = 'avg'
TOP_K = 3
DROPOUT = 0.8
FREEZEING_BASE = False
NO_PARTIAL_BN = True
TRANSFER_LEARNING = False
PATH_TO_SAVE_MODEL = '/datasets/smart-vision/models/experiment'
RESUME_EXP_PATH = None

DATASET = 'smart-vision'
NUM_CLASSES = 5
IS_VIDEO_MULTI_LABELS = False
IS_DATA_FROM_VIDEOS = True
DATA_PATH = '/datasets/charades/data/videos/Charades_v1_480'
TRAINING_DATA_LIST = '/datasets/charades/meta-data/charades_classifier-a_stable_balanced_v2.json'
VALIDATION_LIST = '/datasets/charades/meta-data/charades_classifier-a_validation.json'

NUM_CYCLES = 5
LEARNER_CYCLE_LENGTH = 1
LEARNER_CYCLE_MULTIPLIER = 2
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 5e-4
CLIP_GRADIENT = 20.0

NUM_GPUS = 1
BATCH_SIZE = 16
NUM_WORKERS = 8

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default=CONV_NETS)
parser.add_argument('--num_segments', type=int, default=NUM_VID_SEGMENTS)
parser.add_argument('--consensus_type', type=str, default=CONSENSUS_TYPE,
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
parser.add_argument('--k', type=int, default=TOP_K)
parser.add_argument('--dropout', '--do', default=DROPOUT, type=float,
                    metavar='DO', help='dropout ratio (default: 0.8)')
parser.add_argument('--freeze_base', default=FREEZEING_BASE, action="store_true")
parser.add_argument('--no_partial_bn', '--npb', default=NO_PARTIAL_BN, action="store_true")
parser.add_argument('--transfer_learning', default=TRANSFER_LEARNING, action="store_true")
parser.add_argument('--model_path', type=str,
                    default=PATH_TO_SAVE_MODEL)
parser.add_argument('--resume', type=str,
                    default=RESUME_EXP_PATH)

# ========================= Dataset Configs =========================
parser.add_argument('--dataset', type=str, default=DATASET)
parser.add_argument('--num_class', default=NUM_CLASSES, type=int,
                    help='number of total classes')
parser.add_argument('--multi_label', default=IS_VIDEO_MULTI_LABELS, action="store_true")
parser.add_argument('--from_videos', default=IS_DATA_FROM_VIDEOS, action="store_true")
parser.add_argument('--data_path', type=str,
                    default=DATA_PATH)
parser.add_argument('--train_list', type=str,
                    default=TRAINING_DATA_LIST)
parser.add_argument('--val_list', type=str,
                    default=VALIDATION_LIST)

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', '--num_cycles', default=NUM_CYCLES, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--cycle_length', default=LEARNER_CYCLE_LENGTH, type=int,
                    help='learning rate schedule restart period')
parser.add_argument('--cycle_mult', default=LEARNER_CYCLE_MULTIPLIER, type=int,
                    help='learning rate schedule restart period')
parser.add_argument('--lr', '--learning-rate', default=LEARNING_RATE, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=WEIGHT_DECAY, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip_gradient', '--gd', default=CLIP_GRADIENT, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')

# ========================= Runtime Configs ==========================
parser.add_argument('--gpus', type=int, default=NUM_GPUS)
parser.add_argument('-b', '--batch-size', default=BATCH_SIZE, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=NUM_WORKERS, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

# Not being used for now...
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--modality', type=str, default='RGBDiff')
