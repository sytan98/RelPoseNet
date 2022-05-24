import random
from os import path as osp
from collections import defaultdict
from PIL import Image
import torch
from typing_extensions import Literal

############################ Train Set ########################################
airsim_config = Literal["normal", "accel", "imu"]
class AirSimRelPoseDataset(object):
    def __init__(self, cfg, airsim_config: airsim_config, split='train',transforms=None):
        self.cfg = cfg
        self.airsim_config = airsim_config
        self.split = split
        self.transforms = transforms
        self.fnames1, self.fnames2, self.imu_input_accel, self.imu_input, self.t_gt, self.q_gt, self.t_imu_gt, self.q_imu_gt = self._read_pairs_txt()

    def _read_pairs_txt(self):
        fnames1, fnames2, imu_input_accel, imu_input, t_gt, q_gt, t_imu_gt, q_imu_gt = [], [], [], [], [], [], [], []

        data_params = self.cfg.data_params

        pairs_txt = data_params.train_pairs_fname if self.split == 'train' else data_params.val_pairs_fname
        with open(pairs_txt, 'r') as f:
            for line in f.readlines()[1::]:
                chunks = line.rstrip().split(' ')
                fnames1.append(osp.join(data_params.img_dir, chunks[0]))
                fnames2.append(osp.join(data_params.img_dir, chunks[1]))
                imu_input_accel.append(torch.FloatTensor([float(chunks[2]), float(chunks[3]), float(chunks[4]), 
                                                          float(chunks[5]), float(chunks[6]), float(chunks[7]), 
                                                          float(chunks[8])]))
                imu_input.append(torch.FloatTensor([float(chunks[2]), float(chunks[3]), float(chunks[4]), 
                                                    float(chunks[5]), float(chunks[6]), float(chunks[7]), 
                                                    float(chunks[8]),
                                                    float(chunks[9]), float(chunks[10]), float(chunks[11])]))
                t_gt.append(torch.FloatTensor([float(chunks[12]), float(chunks[13]), float(chunks[14])]))
                q_gt.append(torch.FloatTensor([float(chunks[15]),
                                               float(chunks[16]),
                                               float(chunks[17]),
                                               float(chunks[18])]))
                t_imu_gt.append(torch.FloatTensor([float(chunks[19]), float(chunks[20]), float(chunks[21]) ]))
                q_imu_gt.append(torch.FloatTensor([float(chunks[22]), float(chunks[23]), float(chunks[24]), float(chunks[25])]))
        # print(len(fnames1), len(fnames2))
        return fnames1, fnames2, imu_input_accel, imu_input, t_gt, q_gt, t_imu_gt, q_imu_gt

    def __getitem__(self, item):
        img1 = Image.open(self.fnames1[item]).convert('RGB')
        img2 = Image.open(self.fnames2[item]).convert('RGB')
        imu_input_accel = self.imu_input_accel[item]
        imu_input = self.imu_input[item]
        t_gt = self.t_gt[item]
        q_gt = self.q_gt[item]
        t_imu_gt = self.t_imu_gt[item]
        q_imu_gt = self.q_imu_gt[item]

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        if self.airsim_config == "normal":
            # randomly flip images in an image pair
            if random.uniform(0, 1) > 0.5 and self.split == "train":
                img1, img2 = img2, img1
                t_gt = -self.t_gt[item]
                q_gt = torch.FloatTensor([q_gt[0], -q_gt[1], -q_gt[2], -q_gt[3]])

            return {'img1': img1,
                    'img2': img2,
                    't_gt': t_gt,
                    'q_gt': q_gt}

        elif self.airsim_config == "accel":
            return {'img1': img1,
                    'img2': img2,
                    'imu' : imu_input_accel,
                    't_gt': t_gt,
                    'q_gt': q_gt,
                    't_imu_gt': t_imu_gt}
        else:
            return {'img1': img1,
                    'img2': img2,
                    'imu' : imu_input,
                    't_gt': t_gt,
                    'q_gt': q_gt,
                    't_imu_gt': t_imu_gt,
                    'q_imu_gt': q_imu_gt}

    def __len__(self):
        return len(self.fnames1)

class SevenScenesRelPoseDataset(object):
    def __init__(self, cfg, split='train', transforms=None):
        self.cfg = cfg
        self.split = split
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)
        for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
            self.scenes_dict[i] = scene

        self.fnames1, self.fnames2, self.t_gt, self.q_gt = self._read_pairs_txt()

    def _read_pairs_txt(self):
        fnames1, fnames2, t_gt, q_gt = [], [], [], []

        data_params = self.cfg.data_params

        pairs_txt = data_params.train_pairs_fname if self.split == 'train' else data_params.val_pairs_fname
        with open(pairs_txt, 'r') as f:
            for line in f:
                chunks = line.rstrip().split(' ')
                scene_id = int(chunks[2])
                if self.scenes_dict[scene_id] == 'heads':
                    fnames1.append(osp.join(data_params.img_dir, self.scenes_dict[scene_id], chunks[0][1:]))
                    fnames2.append(osp.join(data_params.img_dir, self.scenes_dict[scene_id], chunks[1][1:]))

                    t_gt.append(torch.FloatTensor([float(chunks[3]), float(chunks[4]), float(chunks[5])]))
                    q_gt.append(torch.FloatTensor([float(chunks[6]),
                                                float(chunks[7]),
                                                float(chunks[8]),
                                                float(chunks[9])]))
        # print(len(fnames1), len(fnames2))
        return fnames1, fnames2, t_gt, q_gt

    def __getitem__(self, item):
        img1 = Image.open(self.fnames1[item]).convert('RGB')
        img2 = Image.open(self.fnames2[item]).convert('RGB')
        t_gt = self.t_gt[item]
        q_gt = self.q_gt[item]

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        # randomly flip images in an image pair
        if random.uniform(0, 1) > 0.5:
            img1, img2 = img2, img1
            t_gt = -self.t_gt[item]
            q_gt = torch.FloatTensor([q_gt[0], -q_gt[1], -q_gt[2], -q_gt[3]])

        return {'img1': img1,
                'img2': img2,
                't_gt': t_gt,
                'q_gt': q_gt}

    def __len__(self):
        return len(self.fnames1)

############################ Test Set ########################################
class SevenScenesTestDataset(object):
    def __init__(self, experiment_cfg, transforms=None):
        self.experiment_cfg = experiment_cfg
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)
        for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
            self.scenes_dict[i] = scene

        self.fnames1, self.fnames2 = self._read_pairs_txt()

    def _read_pairs_txt(self):
        fnames1, fnames2 = [], []

        pairs_txt = self.experiment_cfg.paths.test_pairs_fname
        img_dir = self.experiment_cfg.paths.img_path
        with open(pairs_txt, 'r') as f:
            for line in f:
                chunks = line.rstrip().split(' ')
                scene_id1 = int(chunks[2])
                scene_id2 = int(chunks[3])
                fnames1.append(osp.join(img_dir, self.scenes_dict[scene_id2], chunks[1][1:]))
                fnames2.append(osp.join(img_dir, self.scenes_dict[scene_id1], chunks[0][1:]))

        return fnames1, fnames2

    def __getitem__(self, item):
        img1 = Image.open(self.fnames1[item]).convert('RGB')
        img2 = Image.open(self.fnames2[item]).convert('RGB')

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return {'img1': img1,
                'img2': img2,
                }

    def __len__(self):
        return len(self.fnames1)

class AirSimTestDataset(object):
    def __init__(self, experiment_cfg, airsim_config: airsim_config, transforms=None):
        self.experiment_cfg = experiment_cfg
        self.airsim_config = airsim_config
        self.transforms = transforms
        self.fnames1, self.fnames2, self.imu_input_accel, self.imu_input = self._read_pairs_txt()

    def _read_pairs_txt(self):
        fnames1, fnames2, imu_input_accel, imu_input = [], [], [], []

        pairs_txt = self.experiment_cfg.paths.test_pairs_fname
        img_dir = self.experiment_cfg.paths.img_path
        with open(pairs_txt, 'r') as f:
            for line in f.readlines()[1::]:
                chunks = line.rstrip().split(' ')
                fnames1.append(osp.join(img_dir, chunks[0]))
                fnames2.append(osp.join(img_dir, chunks[1]))      
                imu_input_accel.append(torch.FloatTensor([float(chunks[2]), float(chunks[3]), float(chunks[4]), 
                                            float(chunks[5]), float(chunks[6]), float(chunks[7]), 
                                            float(chunks[8])]))
                imu_input.append(torch.FloatTensor([float(chunks[2]), float(chunks[3]), float(chunks[4]), 
                                                    float(chunks[5]), float(chunks[6]), float(chunks[7]), 
                                                    float(chunks[8]),
                                                    float(chunks[9]), float(chunks[10]), float(chunks[11])]))

        # print(len(fnames1), len(fnames2))
        return fnames1, fnames2, imu_input_accel, imu_input

    def __getitem__(self, item):
        img1 = Image.open(self.fnames1[item]).convert('RGB')
        img2 = Image.open(self.fnames2[item]).convert('RGB')
        imu_input_accel = self.imu_input_accel[item]
        imu_input = self.imu_input[item]

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
        if self.airsim_config == "normal":
            return {'img1': img1,
                    'img2': img2}
        elif self.airsim_config == "accel":
            return {'img1': img1,
                    'img2': img2,
                    'imu' : imu_input_accel}
        else:
            return {'img1': img1,
                    'img2': img2,
                    'imu' : imu_input}
    def __len__(self):
        return len(self.fnames1)