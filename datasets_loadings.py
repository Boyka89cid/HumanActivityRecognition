import torch.utils.data as data
from PIL import Image
import pims
import os.path
import numpy as np
from numpy.random import randint
import gc
import os
import json


class ImageSequenceRecord(object):
    def __init__(self, row, data_path):
        self._data = row
        self.data_path = data_path

    @property
    def path(self):
        class_name = self._data[0].strip().split('_')[1]
        return os.path.join(self.data_path, class_name, self._data[0])

    @property
    def num_frames(self):
        return int(self._data[2])

    @property
    def label(self):
        return int(self._data[1])


class VideoRecord(object):
    # static variable
    def __init__(self, row, data_path, multi_label):
        self._data = row
        self.data_path = data_path
        self.ava_vlog_path = ''
        self.multi_label = multi_label

    @property
    def path(self):
        if self._data['dataset'] == "charades":
            video_name = self._data['video']
            video_name = video_name if video_name.endswith('.mp4') else video_name + '.mp4'
            return os.path.join(self.data_path, video_name)
        elif self._data['dataset'] == "ava":
            return os.path.join(self.ava_vlog_path + self._data['dataset'], "clips", self._data['video'] + ".mp4")
        elif self._data['dataset'] == "vlog":
            return os.path.join(self.ava_vlog_path + self._data['dataset'], "clips", self._data['video'] + "clip.mp4")

    @property
    def label(self):
        labels = self._data['labels']
        if not self.multi_label and isinstance(labels, list):
            return labels[0]
        else:
            return labels

    @property
    def num_frames(self):
        return int(self._data['n_frames']) - 5

    @property
    def start_frame(self):
        return int(np.ceil(self._data['frame_rate'] * self._data['start']))

    @property
    def end_frame(self):
        return min(int(np.floor(self._data['frame_rate'] * self._data['end'])), self._data['n_frames'] - 5)


class ListIndices(object):
    """docstring for ListIndices"""

    def __init__(self, label):
        super(ListIndices, self).__init__()
        self.label = label
        self.cls_inds = []

    def find_cls_inds(self, ls_labels):
        ind = 0
        for eachls in ls_labels:
            if self.label in eachls:
                self.cls_inds += [ind]
            ind = ind + 1

    def getlabel(self):
        return self.label

    def get_cls_inds(self):
        return self.cls_inds


class TwoStreamTSNDataset(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None, split=1, arguments=None,
                 force_grayscale=False, random_shift=True, test_mode=False, basic=None):
        self.args = arguments
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.stream_one_transform, self.stream_two_transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.split = split
        self.basic = basic

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image_from_video(self, video_frame):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            result = [Image.fromarray(video_frame, 'RGB')]
            del video_frame
            return result

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(directory + self.image_tmpl.format(idx))]
        elif self.modality == 'Flow':
            x_img = Image.open(directory + self.image_tmpl.format(idx)).convert('L')
            y_img = Image.open(directory + self.image_tmpl.format(idx)).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        if not self.args.from_videos:
            self.video_list = [ImageSequenceRecord(x.strip().split(','), self.args.data_path) for x in
                               open(self.list_file)]
        else:
            self.video_list = [VideoRecord(json.loads(x), self.args.data_path, self.args.multi_label) for x in
                               open(self.list_file, 'r') if
                               self.split == int(json.loads(x)['split'])]

    def _sample_indices(self, record):
        if not self.args.from_videos:
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif average_duration == 0:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1
        else:
            average_duration = (record.end_frame - record.start_frame - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                frames_range = [record.start_frame, record.end_frame - self.new_length + 1]
                percentiles = np.array(range(self.num_segments)) * 100 / self.num_segments
                indexes = np.percentile(frames_range, percentiles).astype(int)
                indexes += np.random.randint(average_duration, size=self.num_segments)
            elif average_duration == 0 and (record.end_frame - record.start_frame - self.new_length + 1) > 0:
                indexes = record.start_frame + np.sort(
                    randint(record.end_frame - record.start_frame - self.new_length + 1, size=self.num_segments))
            else:
                indexes = record.start_frame + np.zeros((self.num_segments,))
            return indexes

    def _get_val_indices(self, record):
        if not self.args.from_videos:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.end_frame - record.start_frame - self.new_length + 1) / float(self.num_segments)
                indexes = record.start_frame + np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                indexes = record.start_frame + np.zeros((self.num_segments,))

            return indexes

    def _get_test_indices(self, record):

        if not self.args.from_videos:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (record.end_frame - record.start_frame - self.new_length + 1) / float(self.num_segments)
            indexes = record.start_frame + np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

            return indexes + 1

    def to_multihot_encoder(self, record):
        labels_list = record.label

        if not labels_list:
            return np.zeros(self.args.num_class)
        else:
            labels_np = np.array(labels_list).reshape(-1)
            one_hot = np.eye(self.args.num_class)[labels_np]
            return sum(one_hot[:])

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        stream_one_images = list()
        stream_two_images = list()
        if self.args.from_videos:
            video = pims.PyAVVideoReader(record.path)
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                if not self.args.from_videos:
                    seg_imgs = self._load_image(record.path, p + i)
                else:
                    seg_imgs = self._load_image_from_video(video[p + i])
                if i == 2:
                    stream_one_images.extend(seg_imgs)
                stream_two_images.extend(seg_imgs)

        stream_one_aug = self.stream_one_transform(stream_one_images)
        stream_two_aug = self.stream_two_transform(stream_two_images)
        # stream_one_aug = to_np(stream_one_aug)
        # stream_two_aug = to_np(stream_two_aug)

        if self.args.multi_label:
            multihot_encoded = self.to_multihot_encoder(record).astype(np.float32)

        labels = multihot_encoded if self.args.multi_label else record.label

        if self.args.from_videos:
            del video
            del stream_one_images
            del stream_two_images

            gc.collect()

        return stream_one_aug, stream_two_aug, labels

    def __len__(self):
        return len(self.video_list)


class ModelData(object):

    def __init__(self, path, trn_dl, val_dl, test_dl=None):
        self.path, self.trn_dl, self.val_dl, self.test_dl = path, trn_dl, val_dl, test_dl

    @classmethod
    def from_dls(cls, path, trn_dl, val_dl, test_dl=None):
        # trn_dl,val_dl = DataLoader(trn_dl),DataLoader(val_dl)
        # if test_dl: test_dl = DataLoader(test_dl)
        return cls(path, trn_dl, val_dl, test_dl)

    @property
    def is_reg(self): return self.trn_ds.is_reg

    @property
    def is_multi(self): return self.trn_ds.is_multi

    @property
    def trn_ds(self): return self.trn_dl.dataset

    @property
    def val_ds(self): return self.val_dl.dataset

    @property
    def test_ds(self): return self.test_dl.dataset

    @property
    def trn_y(self): return self.trn_ds.y

    @property
    def val_y(self): return self.val_ds.y
