from logging import raiseExceptions
import os
from os import path as osp
import numpy as np
import time
from tqdm import tqdm
import torch
from relposenet.criterion import RelPoseCriterionWithAccel, RelPoseCriterionWithIMU
from relposenet.model import RelPoseNetLarger, RelPoseNetWithAccel, RelPoseNetWithIMU
from tensorboardX import SummaryWriter
from relposenet.model import RelPoseNet
from relposenet.dataset import AirSimRelPoseDataset 
from relposenet.augmentations import get_augmentations
from relposenet.criterion import RelPoseCriterion
from relposenet.utils import cycle, set_seed, cal_quat_angle_error
from abc import ABC, abstractmethod 

class PipelineBase(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        cfg_model = self.cfg.model_params
        self.cfg_model_mode = self.cfg.model_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        set_seed(self.cfg.seed)

        # initialize dataloaders
        self.train_loader, self.val_loader = self._init_dataloaders()
        self.train_loader_iterator = iter(self.train_loader)

        if self.cfg_model_mode == "accel":
            self.model = RelPoseNetWithAccel(cfg_model).to(self.device)
            # Criterion
            self.criterion = RelPoseCriterionWithAccel(self.cfg.train_params.alpha, 
                                                       self.cfg.train_params.pose_weight, 
                                                       self.cfg.train_params.imu_weight, 
                                                       self.cfg.train_params.loss).to(self.device)
        elif self.cfg_model_mode == "imu":
            self.model = RelPoseNetWithIMU(cfg_model).to(self.device)
            # Criterion
            self.criterion = RelPoseCriterionWithIMU(self.cfg.train_params.alpha, 
                                                     self.cfg.train_params.pose_weight, 
                                                     self.cfg.train_params.imu_weight, 
                                                     self.cfg.train_params.loss).to(self.device)
        else:
            if self.cfg.model_size == "large":
                self.model = RelPoseNetLarger(cfg_model).to(self.device)
            else:
                self.model = RelPoseNet(cfg_model).to(self.device)
            # Criterion
            self.criterion = RelPoseCriterion(self.cfg.train_params.alpha, 
                                              self.cfg.train_params.loss).to(self.device)

        print(self.model)
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.cfg.train_params.lr)

        # Scheduler
        cfg_scheduler = self.cfg.train_params.scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=cfg_scheduler.lrate_decay_steps,
                                                         gamma=cfg_scheduler.lrate_decay_factor)

        # create writer (logger)
        self.writer = SummaryWriter(self.cfg.output_params.logger_dir)

        self.start_epoch = 0
        self.val_total_loss = 1e6
        if self.cfg.model_params.resume_snapshot:
            self._load_model(self.cfg.model_params.resume_snapshot)

    @abstractmethod
    def _init_dataloaders(self):
        pass

    @abstractmethod   
    def _predict_cam_pose(self, mini_batch):
        pass

    def _save_model(self, epoch, loss_val, best_val=False):
        if not osp.exists(self.cfg.output_params.snapshot_dir):
            os.makedirs(self.cfg.output_params.snapshot_dir)

        fname_out = 'best_val.pth' if best_val else 'snapshot{:06d}.pth'.format(epoch)
        save_path = osp.join(self.cfg.output_params.snapshot_dir, fname_out)
        model_state = self.model.state_dict()
        torch.save({'epoch': epoch,
                    'state_dict': model_state,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'val_loss': loss_val,
                    },
                   save_path)

    def _load_model(self, snapshot):
        data_dict = torch.load(snapshot)
        self.model.load_state_dict(data_dict['state_dict'])
        self.optimizer.load_state_dict(data_dict['optimizer'])
        self.scheduler.load_state_dict(data_dict['scheduler'])
        self.start_epoch = data_dict['epoch']
        if 'val_loss' in data_dict:
            self.val_total_loss = data_dict['val_loss']

    @abstractmethod
    def _train_batch(self):
        pass

    @abstractmethod
    def _validate(self):
        pass

    @abstractmethod
    def run(self):
        pass

class PipelineWithNormal(PipelineBase):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def _init_dataloaders(self):
        cfg_train = self.cfg.train_params

        # get image augmentations
        train_augs, val_augs = get_augmentations()

        train_dataset = AirSimRelPoseDataset(cfg=self.cfg, airsim_config="normal", split='train', transforms=train_augs)

        val_dataset = AirSimRelPoseDataset(cfg=self.cfg, airsim_config="normal", split='val', transforms=val_augs)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg_train.bs,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=cfg_train.n_workers,
                                                   drop_last=True)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=cfg_train.bs,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=cfg_train.n_workers,
                                                 drop_last=True)
        return train_loader, val_loader
    
    def _predict_cam_pose(self, mini_batch):
        q_est, t_est = self.model.forward(mini_batch['img1'].to(self.device),
                                          mini_batch['img2'].to(self.device))
        return q_est, t_est
    
    def _train_batch(self, train_sample):
        q_est, t_est = self._predict_cam_pose(train_sample)

        self.optimizer.zero_grad()

        # compute loss
        loss, t_loss_val, q_loss_val, t_mse_loss_val, q_mse_loss_val = self.criterion(train_sample['q_gt'].to(self.device),
                                                                                      train_sample['t_gt'].to(self.device),
                                                                                      q_est,
                                                                                      t_est)
        loss.backward()

        # update the optimizer
        self.optimizer.step()

        # update the scheduler
        self.scheduler.step()
        return loss.item(), t_loss_val, q_loss_val, t_mse_loss_val, q_mse_loss_val

    def _validate(self):
        self.model.eval()
        loss_total = t_loss_total = q_loss_total = t_mse_loss_total = q_mse_loss_total = 0.
        pos_err = []
        ori_err = []

        with torch.no_grad():
            for val_sample in tqdm(self.val_loader):
                q_est, t_est = self._predict_cam_pose(val_sample)
                # compute loss
                loss, t_loss_val, q_loss_val, t_mse_loss_val, q_mse_loss_val = self.criterion(val_sample['q_gt'].to(self.device),
                                                                                      val_sample['t_gt'].to(self.device),
                                                                                      q_est,
                                                                                      t_est)
                loss_total += loss.item()
                t_loss_total += t_loss_val
                q_loss_total += q_loss_val
                t_mse_loss_total += t_mse_loss_val
                q_mse_loss_total += q_mse_loss_val
                
                t_err = np.linalg.norm(t_est.cpu().numpy() - val_sample['t_gt'].numpy(), axis=1)
                q_err = cal_quat_angle_error(q_est.cpu().numpy(), val_sample['q_gt'].numpy())
                pos_err += list(t_err)
                ori_err += list(q_err)

        err = (np.median(pos_err), np.median(ori_err))
        print(f'Accuracy: ({err[0]:.2f}m, {err[1]:.2f}deg)')    

        avg_total_loss = loss_total / len(self.val_loader)
        avg_t_loss = t_loss_total / len(self.val_loader)
        avg_q_loss = q_loss_total / len(self.val_loader)
        avg_t_mse_loss = t_mse_loss_total / len(self.val_loader)
        avg_q_mse_loss = q_mse_loss_total / len(self.val_loader)

        self.model.train()

        return avg_total_loss, avg_t_loss, avg_q_loss, avg_t_mse_loss, avg_q_mse_loss

    def run(self):
        train_start_time = time.time()
        train_log_iter_time = time.time()
        if self.cfg.mode == "train":
            print('Start normal model training', self.start_epoch)
            for epoch in range(self.start_epoch, self.cfg.train_params.epoch + 1):
                print("epoch", epoch)
                running_loss = 0.0
                for step, train_sample in enumerate(self.train_loader):
                    train_loss_batch, t_loss, q_loss, t_mse_loss, q_mse_loss,  = self._train_batch(train_sample)
                    running_loss += train_loss_batch

                if epoch % self.cfg.output_params.log_scalar_interval == 0:
                    print(f'Elapsed time [min] for {self.cfg.output_params.log_scalar_interval} epochs: '
                        f'{(time.time() - train_log_iter_time) / 60.}')
                    train_log_iter_time = time.time()
                    print(f'Epoch {epoch}, Train_total_loss {running_loss/len(self.train_loader)}, current lr {self.scheduler.get_last_lr()}')
                    self.writer.add_scalar('Train_total_loss_batch', running_loss/len(self.train_loader), epoch)
                    self.writer.add_scalar('Train_t_loss', t_loss, epoch)
                    self.writer.add_scalar('Train_q_loss', q_loss, epoch)
                    self.writer.add_scalar('Train_t_mse_loss', t_mse_loss, epoch)
                    self.writer.add_scalar('Train_q_mse_loss', q_mse_loss, epoch)

                if epoch % self.cfg.output_params.validate_interval == 0:
                    val_time = time.time()
                    best_val = False
                    val_total_loss, val_t_loss, val_q_loss, val_t_mse_loss, val_q_mse_loss = self._validate()
                    self.writer.add_scalar('Val_total_loss', val_total_loss, epoch)
                    self.writer.add_scalar('Val_total_true_pose_loss', val_t_loss + val_q_loss, epoch)
                    self.writer.add_scalar('Val_t_loss', val_t_loss, epoch)
                    self.writer.add_scalar('Val_q_loss', val_q_loss, epoch)
                    self.writer.add_scalar('Val_t_mse_loss', val_t_mse_loss, epoch)
                    self.writer.add_scalar('Val_q_mse_loss', val_q_mse_loss, epoch)
                    if val_total_loss < self.val_total_loss:
                        self.val_total_loss = val_total_loss
                        best_val = True
                    self._save_model(epoch, val_total_loss, best_val=best_val)
                    print(f'Validation loss: {val_total_loss}, t_loss: {val_t_loss}, q_loss: {val_q_loss}, current lr {self.scheduler.get_last_lr()}')
                    print(f'Elapsed time [min] for validation: {(time.time() - val_time) / 60.}')
                    train_log_iter_time = time.time()

            print(f'Elapsed time for training [min] {(time.time() - train_start_time) / 60.}')
            print('Done')
        elif self.cfg.mode == "val":
            print('Start normal model evaluation')
            val_time = time.time()
            val_total_loss, val_t_loss, val_q_loss, val_t_mse_loss, val_q_mse_loss = self._validate()
            print(f'Validation loss: {val_total_loss}, t_loss: {val_t_loss}, q_loss: {val_q_loss}')
            print(f'Elapsed time [min] for validation: {(time.time() - val_time) / 60.}')
        else:
            raise ValueError("No such mode")
class PipelineWithAccel(PipelineBase):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def _init_dataloaders(self):
        cfg_train = self.cfg.train_params

        # get image augmentations
        train_augs, val_augs = get_augmentations()

        train_dataset = AirSimRelPoseDataset(cfg=self.cfg, airsim_config="accel", split='train', transforms=train_augs)

        val_dataset = AirSimRelPoseDataset(cfg=self.cfg, airsim_config="accel", split='val', transforms=val_augs)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg_train.bs,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=cfg_train.n_workers,
                                                   drop_last=True)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=cfg_train.bs,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=cfg_train.n_workers,
                                                 drop_last=True)
        return train_loader, val_loader
    
    def _predict_cam_pose(self, mini_batch):
        q_est, t_est, t_imu_est = self.model.forward(mini_batch['img1'].to(self.device),
                                          mini_batch['img2'].to(self.device),
                                          mini_batch['imu'].to(self.device))
        return q_est, t_est, t_imu_est
    
    def _train_batch(self, train_sample):
        q_est, t_est, t_imu_est = self._predict_cam_pose(train_sample)

        self.optimizer.zero_grad()

        # compute loss
        losses = self.criterion(train_sample['q_gt'].to(self.device),
                                train_sample['t_gt'].to(self.device),
                                train_sample['t_imu_gt'].to(self.device),
                                q_est,
                                t_est,
                                t_imu_est)

        loss, t_loss_val, q_loss_val, t_imu_loss_val, t_mse_loss_val, q_mse_loss_val, t_mse_imu_loss_val = losses
        loss.backward()

        # update the optimizer
        self.optimizer.step()

        # update the scheduler
        self.scheduler.step()
        return loss.item(), t_loss_val, q_loss_val, t_imu_loss_val, t_mse_loss_val, q_mse_loss_val, t_mse_imu_loss_val

    def _validate(self):
        self.model.eval()
        loss_total = t_loss_total = q_loss_total = t_imu_loss_total = t_mse_loss_total = q_mse_loss_total = t_mse_imu_loss_total = 0.
        pos_err, ori_err = [], []

        with torch.no_grad():
            for val_sample in tqdm(self.val_loader):
                q_est, t_est, t_imu_est = self._predict_cam_pose(val_sample)
                # compute loss
                losses = self.criterion(val_sample['q_gt'].to(self.device),
                                        val_sample['t_gt'].to(self.device),
                                        val_sample['t_imu_gt'].to(self.device),
                                        q_est,
                                        t_est,
                                        t_imu_est)
                loss, t_loss_val, q_loss_val, t_imu_loss_val, t_mse_loss_val, q_mse_loss_val, t_mse_imu_loss_val = losses
                loss_total += loss.item()
                t_loss_total += t_loss_val
                q_loss_total += q_loss_val
                t_imu_loss_total += t_imu_loss_val
                t_mse_loss_total += t_mse_loss_val
                q_mse_loss_total += q_mse_loss_val
                t_mse_imu_loss_total += t_mse_imu_loss_val

                t_err = np.linalg.norm(t_est.cpu().numpy() - val_sample['t_gt'].numpy(), axis=1)
                q_err = cal_quat_angle_error(q_est.cpu().numpy(), val_sample['q_gt'].numpy())
                pos_err += list(t_err)
                ori_err += list(q_err)

        err = (np.median(pos_err), np.median(ori_err))
        print(f'Accuracy: ({err[0]:.2f}m, {err[1]:.2f}deg)')   

        avg_total_loss = loss_total / len(self.val_loader)
        avg_t_loss = t_loss_total / len(self.val_loader)
        avg_q_loss = q_loss_total / len(self.val_loader)
        avg_t_imu_loss = t_imu_loss_total / len(self.val_loader)
        avg_t_mse_loss = t_mse_loss_total / len(self.val_loader)
        avg_q_mse_loss = q_mse_loss_total / len(self.val_loader)
        avg_t_mse_imu_loss = t_mse_imu_loss_total / len(self.val_loader)

        self.model.train()

        return avg_total_loss, avg_t_loss, avg_q_loss, avg_t_imu_loss, avg_t_mse_loss, avg_q_mse_loss, avg_t_mse_imu_loss

    def run(self):
        train_start_time = time.time()
        train_log_iter_time = time.time()
        if self.cfg.mode == "train":
            print('Start Accel model training', self.start_epoch)
            for epoch in range(self.start_epoch, self.cfg.train_params.epoch + 1):
                print("epoch", epoch)
                running_loss = 0.0
                for step, train_sample in enumerate(self.train_loader):
                    train_loss_batch, t_loss, q_loss, t_imu_loss, t_mse_loss, q_mse_loss, t_mse_imu_loss= self._train_batch(train_sample)
                    running_loss += train_loss_batch

                if epoch % self.cfg.output_params.log_scalar_interval == 0:
                    print(f'Elapsed time [min] for {self.cfg.output_params.log_scalar_interval} epochs: '
                        f'{(time.time() - train_log_iter_time) / 60.}')
                    train_log_iter_time = time.time()
                    print(f'Epoch {epoch}, Train_total_loss {running_loss/len(self.train_loader)}')
                    self.writer.add_scalar('Train_total_loss_batch', running_loss/len(self.train_loader), epoch)
                    self.writer.add_scalar('Train_t_loss', t_loss, epoch)
                    self.writer.add_scalar('Train_q_loss', q_loss, epoch)
                    self.writer.add_scalar('Train_t_imu_loss', t_imu_loss, epoch)
                    self.writer.add_scalar('Train_t_mse_loss', t_mse_loss, epoch)
                    self.writer.add_scalar('Train_q_mse_loss', q_mse_loss, epoch)
                    self.writer.add_scalar('Train_t_mse_imu_loss', t_mse_imu_loss, epoch)

                if epoch % self.cfg.output_params.validate_interval == 0:
                    val_time = time.time()
                    best_val = False
                    val_total_loss, val_t_loss, val_q_loss, val_t_imu_loss, val_t_mse_loss, val_q_mse_loss, val_t_mse_imu_loss = self._validate()
                    self.writer.add_scalar('Val_total_loss', val_total_loss, epoch)
                    self.writer.add_scalar('Val_total_true_pose_loss', val_t_loss + val_q_loss, epoch)
                    self.writer.add_scalar('Val_t_loss', val_t_loss, epoch)
                    self.writer.add_scalar('Val_q_loss', val_q_loss, epoch)
                    self.writer.add_scalar('Val_t_imu_loss', val_t_imu_loss, epoch)
                    self.writer.add_scalar('Val_t_mse_loss', val_t_mse_loss, epoch)
                    self.writer.add_scalar('Val_q_mse_loss', val_q_mse_loss, epoch)
                    self.writer.add_scalar('Val_t_mse_imu_loss', val_t_mse_imu_loss, epoch)
                    if val_total_loss < self.val_total_loss:
                        self.val_total_loss = val_total_loss
                        best_val = True
                    self._save_model(epoch, val_total_loss, best_val=best_val)
                    print(f'Validation loss: {val_total_loss}, t_loss: {val_t_loss}, q_loss: {val_q_loss}, t_imu_loss: {val_t_imu_loss}')
                    print(f'Elapsed time [min] for validation: {(time.time() - val_time) / 60.}')
                    train_log_iter_time = time.time()

            print(f'Elapsed time for training [min] {(time.time() - train_start_time) / 60.}')
            print('Done')
        elif self.cfg.mode == "val":
            print('Start accel model evaluation')
            val_time = time.time()
            val_total_loss, val_t_loss, val_q_loss, val_t_imu_loss, val_t_mse_loss, val_q_mse_loss, val_t_mse_imu_loss = self._validate()
            print(f'Validation loss: {val_total_loss}, t_loss: {val_t_loss}, q_loss: {val_q_loss}, t_imu_loss: {val_t_imu_loss}')
            print(f'Elapsed time [min] for validation: {(time.time() - val_time) / 60.}')
        else:
            raise ValueError("No such mode")

class PipelineWithIMU(PipelineBase):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def _init_dataloaders(self):
        cfg_train = self.cfg.train_params

        # get image augmentations
        train_augs, val_augs = get_augmentations()

        train_dataset = AirSimRelPoseDataset(cfg=self.cfg, airsim_config="imu", split='train', transforms=train_augs)

        val_dataset = AirSimRelPoseDataset(cfg=self.cfg, airsim_config="imu", split='val', transforms=val_augs)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg_train.bs,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=cfg_train.n_workers,
                                                   drop_last=True)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=cfg_train.bs,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=cfg_train.n_workers,
                                                 drop_last=True)
        return train_loader, val_loader
    
    def _predict_cam_pose(self, mini_batch):
        q_est, t_est, q_imu_est, t_imu_est = self.model.forward(mini_batch['img1'].to(self.device),
                                          mini_batch['img2'].to(self.device),
                                          mini_batch['imu'].to(self.device))
        return q_est, t_est, q_imu_est, t_imu_est

    def _train_batch(self, train_sample):
        q_est, t_est, q_imu_est, t_imu_est = self._predict_cam_pose(train_sample)

        self.optimizer.zero_grad()

        # compute loss
        losses= self.criterion(train_sample['q_gt'].to(self.device),
                               train_sample['t_gt'].to(self.device),
                               train_sample['q_imu_gt'].to(self.device),
                               train_sample['t_imu_gt'].to(self.device),
                               q_est,
                               t_est,
                               q_imu_est,
                               t_imu_est)
        loss, t_loss_val, q_loss_val, t_imu_loss_val, q_imu_loss_val, t_mse_loss_val, q_mse_loss_val, t_mse_imu_loss_val, q_mse_imu_loss_val = losses
        loss.backward()

        # update the optimizer
        self.optimizer.step()

        # update the scheduler
        self.scheduler.step()
        return loss.item(), t_loss_val, q_loss_val, t_imu_loss_val, q_imu_loss_val, t_mse_loss_val, q_mse_loss_val, t_mse_imu_loss_val, q_mse_imu_loss_val

    def _validate(self):
        self.model.eval()
        loss_total = t_loss_total = q_loss_total = t_imu_loss_total = q_imu_loss_total = t_mse_loss_total = q_mse_loss_total = t_mse_imu_loss_total = q_mse_imu_loss_total = 0.
        pos_err, ori_err = [], []
        with torch.no_grad():
            for val_sample in tqdm(self.val_loader):
                q_est, t_est, q_imu_est, t_imu_est = self._predict_cam_pose(val_sample)
                # compute loss
                losses = self.criterion(val_sample['q_gt'].to(self.device),
                                        val_sample['t_gt'].to(self.device),
                                        val_sample['q_imu_gt'].to(self.device),
                                        val_sample['t_imu_gt'].to(self.device),
                                        q_est,
                                        t_est,
                                        q_imu_est,
                                        t_imu_est)
                loss, t_loss_val, q_loss_val, t_imu_loss_val, q_imu_loss_val, t_mse_loss_val, q_mse_loss_val, t_mse_imu_loss_val, q_mse_imu_loss_val = losses
                loss_total += loss.item()
                t_loss_total += t_loss_val
                q_loss_total += q_loss_val
                t_imu_loss_total += t_imu_loss_val
                q_imu_loss_total += q_imu_loss_val
                t_mse_loss_total += t_mse_loss_val
                q_mse_loss_total += q_mse_loss_val
                t_mse_imu_loss_total += t_mse_imu_loss_val
                q_mse_imu_loss_total += q_mse_imu_loss_val

                t_err = np.linalg.norm(t_est.cpu().numpy() - val_sample['t_gt'].numpy(), axis=1)
                q_err = cal_quat_angle_error(q_est.cpu().numpy(), val_sample['q_gt'].numpy())
                pos_err += list(t_err)
                ori_err += list(q_err)

        err = (np.median(pos_err), np.median(ori_err))
        print(f'Accuracy: ({err[0]:.2f}m, {err[1]:.2f}deg)')   

        avg_total_loss = loss_total / len(self.val_loader)
        avg_t_loss = t_loss_total / len(self.val_loader)
        avg_q_loss = q_loss_total / len(self.val_loader)
        avg_t_imu_loss = t_imu_loss_total / len(self.val_loader)
        avg_q_imu_loss = q_imu_loss_total / len(self.val_loader)
        avg_t_mse_loss = t_mse_loss_total / len(self.val_loader)
        avg_q_mse_loss = q_mse_loss_total / len(self.val_loader)
        avg_t_mse_imu_loss = t_mse_imu_loss_total / len(self.val_loader)
        avg_q_mse_imu_loss = q_mse_imu_loss_total / len(self.val_loader)

        self.model.train()

        return avg_total_loss, avg_t_loss, avg_q_loss, avg_t_imu_loss, avg_q_imu_loss, avg_t_mse_loss, avg_q_mse_loss, avg_t_mse_imu_loss, avg_q_mse_imu_loss
    
    def run(self):
        train_start_time = time.time()
        train_log_iter_time = time.time()
        if self.cfg.mode == "train":
            print('Start imu model training', self.start_epoch)
            for epoch in range(self.start_epoch, self.cfg.train_params.epoch + 1):
                print("epoch", epoch)
                running_loss = 0.0
                for step, train_sample in enumerate(self.train_loader):
                    train_loss_batch, t_loss, q_loss, t_imu_loss, q_imu_loss, t_mse_loss, q_mse_loss, t_mse_imu_loss, q_mse_imu_loss = self._train_batch(train_sample)
                    running_loss += train_loss_batch

                if epoch % self.cfg.output_params.log_scalar_interval == 0:
                    print(f'Elapsed time [min] for {self.cfg.output_params.log_scalar_interval} epochs: '
                        f'{(time.time() - train_log_iter_time) / 60.}')
                    train_log_iter_time = time.time()
                    print(f'Epoch {epoch}, Train_total_loss {running_loss/len(self.train_loader)}')
                    self.writer.add_scalar('Train_total_loss_batch', running_loss/len(self.train_loader), epoch)
                    self.writer.add_scalar('Train_t_loss', t_loss, epoch)
                    self.writer.add_scalar('Train_q_loss', q_loss, epoch)
                    self.writer.add_scalar('Train_t_imu_loss', t_imu_loss, epoch)
                    self.writer.add_scalar('Train_q_imu_loss', q_imu_loss, epoch)
                    self.writer.add_scalar('Train_t_mse_loss', t_mse_loss, epoch)
                    self.writer.add_scalar('Train_q_mse_loss', q_mse_loss, epoch)
                    self.writer.add_scalar('Train_t_mse_imu_loss', t_mse_imu_loss, epoch)
                    self.writer.add_scalar('Train_q_mse_imu_loss', q_mse_imu_loss, epoch)

                if epoch % self.cfg.output_params.validate_interval == 0:
                    val_time = time.time()
                    best_val = False
                    val_total_loss, val_t_loss, val_q_loss, val_t_imu_loss, val_q_imu_loss, val_t_mse_loss, val_q_mse_loss, val_t_mse_imu_loss, val_q_mse_imu_loss = self._validate()
                    self.writer.add_scalar('Val_total_loss', val_total_loss, epoch)
                    self.writer.add_scalar('Val_total_true_pose_loss', val_t_loss + val_q_loss, epoch)
                    self.writer.add_scalar('Val_t_loss', val_t_loss, epoch)
                    self.writer.add_scalar('Val_q_loss', val_q_loss, epoch)
                    self.writer.add_scalar('Val_t_imu_loss', val_t_imu_loss, epoch)
                    self.writer.add_scalar('Val_q_imu_loss', val_q_imu_loss, epoch)
                    self.writer.add_scalar('Val_t_mse_loss', val_t_mse_loss, epoch)
                    self.writer.add_scalar('Val_q_mse_loss', val_q_mse_loss, epoch)
                    self.writer.add_scalar('Val_t_mse_imu_loss', val_t_mse_imu_loss, epoch)
                    self.writer.add_scalar('Val_q_mse_imu_loss', val_q_mse_imu_loss, epoch)
                    if val_total_loss < self.val_total_loss:
                        self.val_total_loss = val_total_loss
                        best_val = True
                    self._save_model(epoch, val_total_loss, best_val=best_val)
                    print(f'Validation loss: {val_total_loss}, t_loss: {val_t_loss}, q_loss: {val_q_loss}, t_imu_loss: {val_t_imu_loss}, q_imu_loss: {val_q_imu_loss}')
                    print(f'Elapsed time [min] for validation: {(time.time() - val_time) / 60.}')
                    train_log_iter_time = time.time()

            print(f'Elapsed time for training [min] {(time.time() - train_start_time) / 60.}')
            print('Done')
        elif self.cfg.mode == "val":
            print('Start imu model evaluation')
            val_time = time.time()
            val_total_loss, val_t_loss, val_q_loss, val_t_imu_loss, val_q_imu_loss, val_t_mse_loss, val_q_mse_loss, val_t_mse_imu_loss, val_q_mse_imu_loss = self._validate()
            print(f'Validation loss: {val_total_loss}, t_loss: {val_t_loss}, q_loss: {val_q_loss}, t_imu_loss: {val_t_imu_loss}, q_imu_loss: {val_q_imu_loss}')
            print(f'Elapsed time [min] for validation: {(time.time() - val_time) / 60.}')
        else:
            raise ValueError("No such mode")