from abc import ABC
import torch.nn as nn


class RelPoseCriterion(nn.Module, ABC):
    def __init__(self, alpha=1, loss="mse"):
        super().__init__()
        self.alpha = alpha
        self.q_mse_loss = nn.MSELoss()
        self.t_mse_loss = nn.MSELoss()
        self.q_loss = nn.MSELoss() if loss == "mse" else nn.L1Loss()
        self.t_loss = nn.MSELoss() if loss == "mse" else nn.L1Loss()

    def forward(self, q_gt, t_gt, q_est, t_est):
        t_loss = self.t_loss(t_est, t_gt)
        q_loss = self.q_loss(q_est, q_gt)

        t_mse_loss = self.t_mse_loss(t_est, t_gt)
        q_mse_loss = self.q_mse_loss(q_est, q_gt)

        loss_total = t_loss + self.alpha * q_loss
        return loss_total, t_loss.item(), q_loss.item(), \
            t_mse_loss.item(), q_mse_loss.item()


class RelPoseCriterionWithAccel(nn.Module, ABC):
    def __init__(self, alpha=1, pose_weight=1, imu_weight=1, loss="mse"):
        super().__init__()
        self.alpha = alpha
        self.pose_weight = pose_weight
        self.imu_weight = imu_weight
        self.q_mse_loss = nn.MSELoss()
        self.t_mse_loss = nn.MSELoss()
        self.t_mse_imu_loss = nn.MSELoss()
        self.q_loss = nn.MSELoss() if loss == "mse" else nn.L1Loss()
        self.t_loss = nn.MSELoss() if loss == "mse" else nn.L1Loss()
        self.t_imu_loss = nn.MSELoss() if loss == "mse" else nn.L1Loss()

    def forward(self, q_gt, t_gt, t_imu_gt, q_est, t_est, t_imu_est):
        t_loss = self.t_loss(t_est, t_gt)
        q_loss = self.q_loss(q_est, q_gt)
        t_imu_loss = self.t_imu_loss(t_imu_est, t_imu_gt)

        t_mse_loss = self.t_mse_loss(t_est, t_gt)
        q_mse_loss = self.q_mse_loss(q_est, q_gt)
        t_mse_imu_loss = self.t_mse_imu_loss(t_imu_est, t_imu_gt)

        loss_total = self.pose_weight * (t_loss + self.alpha * q_loss) + \
                     self.imu_weight * (t_imu_loss)
        return loss_total, t_loss.item(), q_loss.item(), t_imu_loss.item(), \
            t_mse_loss.item(), q_mse_loss.item(), t_mse_imu_loss.item()

class RelPoseCriterionWithIMU(nn.Module, ABC):
    def __init__(self, alpha=1, pose_weight=1, imu_weight=1, loss="mse"):
        super().__init__()
        self.alpha = alpha
        self.pose_weight = pose_weight
        self.imu_weight = imu_weight
        self.q_mse_loss = nn.MSELoss()
        self.t_mse_loss = nn.MSELoss()
        self.t_mse_imu_loss = nn.MSELoss()
        self.q_mse_imu_loss = nn.MSELoss()
        self.q_loss = nn.MSELoss() if loss == "mse" else nn.L1Loss()
        self.t_loss = nn.MSELoss() if loss == "mse" else nn.L1Loss()
        self.t_imu_loss = nn.MSELoss() if loss == "mse" else nn.L1Loss()
        self.q_imu_loss = nn.MSELoss() if loss == "mse" else nn.L1Loss()

    def forward(self, q_gt, t_gt, q_imu_gt, t_imu_gt, q_est, t_est, q_imu_est, t_imu_est):
        t_loss = self.t_loss(t_est, t_gt)
        q_loss = self.q_loss(q_est, q_gt)
        t_imu_loss = self.t_imu_loss(t_imu_est, t_imu_gt)
        q_imu_loss = self.q_imu_loss(q_imu_est, q_imu_gt)

        t_mse_loss = self.t_mse_loss(t_est, t_gt)
        q_mse_loss = self.q_mse_loss(q_est, q_gt)
        t_mse_imu_loss = self.t_mse_imu_loss(t_imu_est, t_imu_gt)
        q_mse_imu_loss = self.q_mse_imu_loss(q_imu_est, q_imu_gt)

        loss_total = self.pose_weight * (t_loss + self.alpha * q_loss) + \
                     self.imu_weight * (t_imu_loss + self.alpha * q_imu_loss)
        return loss_total, t_loss.item(), q_loss.item(), t_imu_loss.item(), q_imu_loss.item(), \
                t_mse_loss.item(), q_mse_loss.item(), t_mse_imu_loss.item(), q_mse_imu_loss.item()
