import torch.nn as nn
import numpy as np
import torch

v1 = np.array([[ 0.2209424, 0.2209424, 0.2209424, 0.9238795 ],
                [ 0.2209424, 0.2209424, 0.2209424, 0.9238795 ]])
v2 = np.array([[ 0.2439988, 0.2439988, 0.2439988, 0.9063078 ],
                [ 0.2209424, 0.2209424, 0.2209424, 0.9238795 ]])
v1_tensor = torch.from_numpy(v1)
v2_tensor = torch.from_numpy(v2)

# mse_loss = nn.MSELoss(reduction='none')
# mae_loss = nn.L1Loss()

# print(v1_tensor - v2_tensor)
# print(mse_loss(v1_tensor, v2_tensor))
# print(mae_loss(v1_tensor, v2_tensor))

def cal_quat_angle_error(label, pred):
    if len(label.shape) == 1:
        label = np.expand_dims(label, axis=0)
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=0)
    q1 = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    q2 = label / np.linalg.norm(label, axis=1, keepdims=True)
    print(q1, q2)
    print(np.multiply(q1,q2))
    print(np.sum(np.multiply(q1,q2), axis=1, keepdims=True))
    d = np.abs(np.sum(np.multiply(q1,q2), axis=1, keepdims=True)) # Here we have abs()

    d = np.clip(d, a_min=-1, a_max=1)
    error = 2 * np.degrees(np.arccos(d))
    return error

print(cal_quat_angle_error(v1, v2))