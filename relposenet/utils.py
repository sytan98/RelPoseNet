import numpy as np
import random
import torch
from scipy.spatial.transform import Rotation as R

def set_seed(seed):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def obtain_relative_pose(absolute_pose_c1, absolute_pose_c2):
	xyz1, wxyz1 = absolute_pose_c1[:3], absolute_pose_c1[3:]
	xyz2, wxyz2 = absolute_pose_c2[:3], absolute_pose_c2[3:]

	# Airsim records quartenion as wxyz but scipy takes in xyzw format
	rot_mat1 = R.from_quat(np.hstack([wxyz1[1:], wxyz1[0]])).as_matrix()
	rot_mat2 = R.from_quat(np.hstack([wxyz2[1:], wxyz2[0]])).as_matrix()

	# Obtain Relative poses
	relative_t = xyz2 - xyz1 
	relative_q = R.from_matrix(np.matmul(np.transpose(rot_mat1), rot_mat2)).as_quat()
	relative_q = np.hstack([relative_q[3:], relative_q[:3]]) # Convert to wxyz format

	return relative_t, relative_q

def obtain_absolute_pose(absolute_pose_c2, relative_pose):
	xyz2, wpqr2 = absolute_pose_c2[:3], absolute_pose_c2[3:]
	xyz_rel, wpqr_rel = relative_pose[:3], relative_pose[3:]
	rot_mat2 = R.from_quat(np.hstack([wpqr2[1:], wpqr2[0]])).as_matrix()
	rot_mat_rel = R.from_quat(np.hstack([wpqr_rel[1:], wpqr_rel[0]])).as_matrix()
	t1 = xyz2 - xyz_rel 
	r1 = R.from_matrix(np.matmul(rot_mat2,np.linalg.inv(rot_mat_rel))).as_quat()
	r1 = np.hstack([r1[3:], r1[:3]])
	return t1, r1