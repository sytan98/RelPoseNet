import os
from os import path as osp
import numpy as np
from tqdm import tqdm
import torch
from experiments.service.benchmark_base import Benchmark
from relposenet.criterion import RelPoseCriterion
from relposenet.dataset import AirSimTestDataset, AirSimRelPoseDataset 
from relposenet.augmentations import get_augmentations
from relposenet.model import RelPoseNet, RelPoseNetWithIMU
from relposenet.utils import cal_quat_angle_error

class AirSimBenchmark(Benchmark):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.dataloader = self._init_dataloader()
        self.model = self._load_model_relposenet().to(self.device)
        self.criterion = RelPoseCriterion().to(self.device)
    def _init_dataloader(self):
        experiment_cfg = self.cfg.experiment.experiment_params

        # define test augmentations
        _, eval_aug = get_augmentations()

        # test dataset
        dataset = AirSimRelPoseDataset(cfg=self.cfg, airsim_config="normal", split='val', transforms=eval_aug)

        # define a dataloader
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=experiment_cfg.bs,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=experiment_cfg.n_workers,
                                                 drop_last=False)

        return dataloader

    def _load_model_relposenet(self):
        print(f'Loading RelPoseNet model...')
        model_params_cfg = self.cfg.model.model_params
        model = RelPoseNet(model_params_cfg)

        data_dict = torch.load(model_params_cfg.snapshot)
        model.load_state_dict(data_dict['state_dict'])
        print(f'Loading RelPoseNet model... Done!')
        return model.eval()

    def evaluate(self):
        q_est_all, t_est_all = [], []
        loss_total = t_loss_total = q_loss_total = 0.
        pos_err = []
        ori_err = []
        print(f'Evaluate on the dataset...')
        i = 0
        with torch.no_grad():
            for data_batch in tqdm(self.dataloader):
                q_est, t_est = self.model(data_batch['img1'].to(self.device),
                                          data_batch['img2'].to(self.device))
                loss, t_loss_val, q_loss_val, _, _ = self.criterion(data_batch['q_gt'].to(self.device),
                                                                    data_batch['t_gt'].to(self.device),
                                                                    q_est,
                                                                    t_est)
                loss_total += loss.item()
                t_loss_total += t_loss_val
                q_loss_total += q_loss_val
                q_est_all.append(q_est)
                t_est_all.append(t_est)
                t_err = np.linalg.norm(t_est.cpu().numpy() - data_batch['t_gt'].numpy(), axis=1)
                q_err = cal_quat_angle_error(q_est.cpu().numpy(), data_batch['q_gt'].numpy())
                pos_err += list(t_err)
                ori_err += list(q_err)

        err = (np.median(pos_err), np.median(ori_err))
        print(f'Accuracy: ({err[0]:.2f}m, {err[1]:.2f}deg)')    

        avg_total_loss = loss_total / len(self.dataloader)
        avg_t_loss = t_loss_total / len(self.dataloader)
        avg_q_loss = q_loss_total / len(self.dataloader)
        print(f'Validation loss: {avg_total_loss}, t_loss: {avg_t_loss}, q_loss: {avg_q_loss}')

        q_est_all = torch.cat(q_est_all).cpu().numpy()
        t_est_all = torch.cat(t_est_all).cpu().numpy()

        print(f'Write the estimates to a text file')
        experiment_cfg = self.cfg.experiment.experiment_params

        if not osp.exists(experiment_cfg.output.home_dir):
            os.makedirs(experiment_cfg.output.home_dir)

        with open(experiment_cfg.output.res_txt_fname, 'w') as f:
            for q_est, t_est in zip(q_est_all, t_est_all):
                f.write(f"{q_est[0]} {q_est[1]} {q_est[2]} {q_est[3]} {t_est[0]} {t_est[1]} {t_est[2]}\n")

        print(f'Done')

class AirSimWithIMUBenchmark(Benchmark):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.dataloader = self._init_dataloader()
        self.model = self._load_model_relposenet().to(self.device)

    def _init_dataloader(self):
        experiment_cfg = self.cfg.experiment.experiment_params

        # define test augmentations
        _, eval_aug = get_augmentations()

        # test dataset
        dataset = AirSimTestDataset(experiment_cfg, "imu", eval_aug)

        # define a dataloader
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=experiment_cfg.bs,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=experiment_cfg.n_workers,
                                                 drop_last=False)

        return dataloader

    def _load_model_relposenet(self):
        print(f'Loading RelPoseNet model...')
        model_params_cfg = self.cfg.model.model_params
        model = RelPoseNet(model_params_cfg)

        data_dict = torch.load(model_params_cfg.snapshot)
        model.load_state_dict(data_dict['state_dict'])
        print(f'Loading RelPoseNet model... Done!')
        return model.eval()

    def evaluate(self):
        # q_est_all, t_est_all, q_imu_est_all, t_imu_est_all = [], [], [], []
        # print(f'Evaluate on the dataset...')
        # with torch.no_grad():
        #     for data_batch in tqdm(self.dataloader):
        #         q_est, t_est, q_imu_est, t_imu_est = self.model(data_batch['img1'].to(self.device),
        #                                                         data_batch['img2'].to(self.device),
        #                                                         data_batch['imu'].to(self.device))

        #         q_est_all.append(q_est)
        #         t_est_all.append(t_est)
        #         q_imu_est_all.append(q_imu_est)
        #         t_imu_est_all.append(t_imu_est)

        # q_est_all = torch.cat(q_est_all).cpu().numpy()
        # t_est_all = torch.cat(t_est_all).cpu().numpy()
        q_est_all, t_est_all= [], []
        print(f'Evaluate on the dataset...')
        with torch.no_grad():
            for data_batch in tqdm(self.dataloader):
                q_est, t_est, = self.model(data_batch['img1'].to(self.device),
                                            data_batch['img2'].to(self.device))

                q_est_all.append(q_est)
                t_est_all.append(t_est)

        q_est_all = torch.cat(q_est_all).cpu().numpy()
        t_est_all = torch.cat(t_est_all).cpu().numpy()
        print(f'Write the estimates to a text file')
        experiment_cfg = self.cfg.experiment.experiment_params

        if not osp.exists(experiment_cfg.output.home_dir):
            os.makedirs(experiment_cfg.output.home_dir)

        with open(experiment_cfg.output.res_txt_fname, 'w') as f:
            for q_est, t_est in zip(q_est_all, t_est_all):
                f.write(f"{q_est[0]} {q_est[1]} {q_est[2]} {q_est[3]} {t_est[0]} {t_est[1]} {t_est[2]}\n")

        print(f'Done')
