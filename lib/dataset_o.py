import os
import cv2
import math
import datetime
import time
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from numba import jit

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
from models.superpoint import SuperPoint
from models.utils import frame2tensor

from lib.utils import (
    preprocess_image,
    grid_positions,
    upscale_positions,
    downscale_positions,
    savefig,
    imshow_image
)
from lib.exceptions import NoGradientError, EmptyTensorError

matplotlib.use('Agg')

def draw_matches(img1, cv_kpts1, img2, cv_kpts2, match_ids, match_color=(0, 255, 0), pt_color=(0, 0, 255)):
    print(f'matches:{len(match_ids[0])},{len(match_ids[1])}')
    img1 = imshow_image(img1.cpu().numpy(), 'torch')
    cv_kpts1 = cv_kpts1.cpu().numpy().T
    img2 = imshow_image(img2.cpu().numpy(), 'torch')
    cv_kpts2 = cv_kpts2.cpu().numpy().T
    good_matches = []
    for id1, id2 in zip(*match_ids):
        match = cv2.DMatch()
        match.queryIdx = id1
        match.trainIdx = id2
        good_matches.append(match)
    mask = np.ones((len(good_matches), ))
    """Draw matches."""
    if type(cv_kpts1) is np.ndarray and type(cv_kpts2) is np.ndarray:
        cv_kpts1 = [cv2.KeyPoint(cv_kpts1[i][0], cv_kpts1[i][1], 1) for i in range(cv_kpts1.shape[0])]
        cv_kpts2 = [cv2.KeyPoint(cv_kpts2[i][0], cv_kpts2[i][1], 1) for i in range(cv_kpts2.shape[0])]
    display = cv2.drawMatches(img1, cv_kpts1, img2, cv_kpts2, good_matches, None, matchColor=match_color, singlePointColor=pt_color, matchesMask=mask.ravel().tolist(), flags=4)
    cv2.imwrite('match_vis_tmp.png', display)
    return display

@jit
def DistanceValid(P1, P2):
    # 自定义匹配条件：世界系欧几里得距离<0.05米
    return (P1[0] - P2[0]).square() + (P1[1] - P2[1]).square() + (P1[2] - P2[2]).square() < 0.0025

@jit
def ComputeMatches(keypoints1, keypoints2, id1s, id2s, XYZ1, XYZ2):
    matches_ids1 = []
    matches_ids2 = []
    flag1 = [1] * keypoints1.shape[1]
    flag2 = [1] * keypoints2.shape[1]
    # 寻找所有匹配
    # 每个点最多匹配一次
    # start_time = time.time()
    for i1 in range(id1s.shape[0]):
        id1 = int(id1s[i1].item())
        for i2 in range(id2s.shape[0]):
            id2 = int(id2s[i2].item())
            if (flag2[id2] and DistanceValid(XYZ1[:, i1], XYZ2[:, i2])):
                matches_ids1.append(id1)
                matches_ids2.append(id2)
                flag1[id1] = 0
                flag2[id2] = 0
            if (flag1[id1] == 0):
                break
        
    print('keypoints, matches shape:', keypoints1.shape, keypoints2.shape, len(matches_ids1))
    # 添加不存在的匹配（SuperGlue训练代码要求）
    for i in range(keypoints1.shape[1]):
        if flag1[i]:
            matches_ids1.append(i)
            matches_ids2.append(keypoints2.shape[1])
    for i in range(keypoints2.shape[1]):
        if flag2[i]:
            matches_ids1.append(keypoints1.shape[1])
            matches_ids2.append(i)
    
    return matches_ids1, matches_ids2


class MegaDepthDataset(Dataset):
    def __init__(
            self,
            nfeatures=1024,
            scene_list_path='./megadepth_utils/train_scenes.txt',
            scene_info_path='/home/liuy/data/megadepth/output',
            base_path='/home/liuy/data/megadepth',
            train=True,
            preprocessing=None,
            min_overlap_ratio=.5,
            max_overlap_ratio=1,
            max_scale_ratio=np.inf,
            pairs_per_scene=100,
            image_size=256
    ):
        self.scenes = []
        with open(scene_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.scenes.append(line.strip('\n'))

        self.scene_info_path = scene_info_path
        self.base_path = base_path

        self.train = train

        self.preprocessing = preprocessing

        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.max_scale_ratio = max_scale_ratio

        self.pairs_per_scene = pairs_per_scene

        self.image_size = image_size

        self.dataset = []

        self.nfeatures = nfeatures
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)

    def build_dataset(self):
        print('build_dataset: begin')
        self.dataset = []
        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)
            print('Building the validation dataset...')
        else:
            print('Building a new training dataset...')
        for scene in tqdm(self.scenes, total=len(self.scenes)):
            scene_info_path = os.path.join(
                self.scene_info_path, '%s.npz' % scene
            )
            if not os.path.exists(scene_info_path):
                continue
            scene_info = np.load(scene_info_path, allow_pickle=True)
            overlap_matrix = scene_info['overlap_matrix']
            scale_ratio_matrix = scene_info['scale_ratio_matrix']

            valid =  np.logical_and(
                np.logical_and(
                    overlap_matrix >= self.min_overlap_ratio,
                    overlap_matrix <= self.max_overlap_ratio
                ),
                scale_ratio_matrix <= self.max_scale_ratio
            )
            
            pairs = np.vstack(np.where(valid))
            try:
                selected_ids = np.random.choice(
                    pairs.shape[1], self.pairs_per_scene
                )
            except:
                continue
            
            image_paths = scene_info['image_paths']
            depth_paths = scene_info['depth_paths']
            points3D_id_to_2D = scene_info['points3D_id_to_2D']
            points3D_id_to_ndepth = scene_info['points3D_id_to_ndepth']
            intrinsics = scene_info['intrinsics']
            poses = scene_info['poses']
            
            for pair_idx in selected_ids:
                idx1 = pairs[0, pair_idx]
                idx2 = pairs[1, pair_idx]
                matches = np.array(list(
                    points3D_id_to_2D[idx1].keys() &
                    points3D_id_to_2D[idx2].keys()
                ))

                # Scale filtering
                matches_nd1 = np.array([points3D_id_to_ndepth[idx1][match] for match in matches])
                matches_nd2 = np.array([points3D_id_to_ndepth[idx2][match] for match in matches])
                scale_ratio = np.maximum(matches_nd1 / matches_nd2, matches_nd2 / matches_nd1)
                matches = matches[np.where(scale_ratio <= self.max_scale_ratio)[0]]
                
                point3D_id = np.random.choice(matches)
                point2D1 = points3D_id_to_2D[idx1][point3D_id]
                point2D2 = points3D_id_to_2D[idx2][point3D_id]
                nd1 = points3D_id_to_ndepth[idx1][point3D_id]
                nd2 = points3D_id_to_ndepth[idx2][point3D_id]
                central_match = np.array([
                    point2D1[1], point2D1[0],
                    point2D2[1], point2D2[0]
                ])
                self.dataset.append({
                    'image_path1': image_paths[idx1],
                    'depth_path1': depth_paths[idx1],
                    'intrinsics1': intrinsics[idx1],
                    'pose1': poses[idx1],
                    'image_path2': image_paths[idx2],
                    'depth_path2': depth_paths[idx2],
                    'intrinsics2': intrinsics[idx2],
                    'pose2': poses[idx2],
                    'central_match': central_match,
                    'scale_ratio': max(nd1 / nd2, nd2 / nd1)
                })
        np.random.shuffle(self.dataset)
        if not self.train:
            np.random.set_state(np_random_state)
        
        print('build_dataset: end')

    def __len__(self):
        return len(self.dataset)
    
    def parse_superpoint_result(self, result):
        keypoints = result['keypoints'][0].cpu()
        scores = result['scores'][0].cpu()
        shape = scores.shape
        scores = scores.reshape(shape[0], 1)
        kp = torch.cat((keypoints, scores), 1)
        kp = kp.detach().numpy()
        descriptors = result['descriptors'][0].cpu()
        descs = descriptors.T
        descs = descs.detach().numpy()
        return kp, descs

    def recover_pair(self, pair_metadata):
        depth_path1 = os.path.join(
            self.base_path, pair_metadata['depth_path1']
        )
        with h5py.File(depth_path1, 'r') as hdf5_file:
            depth1 = np.array(hdf5_file['/depth'])
        assert(np.min(depth1) >= 0)
        image_path1 = os.path.join(
            self.base_path, pair_metadata['image_path1']
        )
        image1 = Image.open(image_path1)
        if image1.mode != 'RGB':
            image1 = image1.convert('RGB')
        image1 = np.array(image1)
        assert(image1.shape[0] == depth1.shape[0] and image1.shape[1] == depth1.shape[1])
        intrinsics1 = pair_metadata['intrinsics1']
        pose1 = pair_metadata['pose1']

        depth_path2 = os.path.join(
            self.base_path, pair_metadata['depth_path2']
        )
        with h5py.File(depth_path2, 'r') as hdf5_file:
            depth2 = np.array(hdf5_file['/depth'])
        assert(np.min(depth2) >= 0)
        image_path2 = os.path.join(
            self.base_path, pair_metadata['image_path2']
        )
        image2 = Image.open(image_path2)
        if image2.mode != 'RGB':
            image2 = image2.convert('RGB')
        image2 = np.array(image2)
        assert(image2.shape[0] == depth2.shape[0] and image2.shape[1] == depth2.shape[1])
        intrinsics2 = pair_metadata['intrinsics2']
        pose2 = pair_metadata['pose2']

        central_match = pair_metadata['central_match']
        image1, bbox1, image2, bbox2 = self.crop(image1, image2, central_match)

        depth1 = depth1[
            bbox1[0] : bbox1[0] + self.image_size,
            bbox1[1] : bbox1[1] + self.image_size
        ]
        depth2 = depth2[
            bbox2[0] : bbox2[0] + self.image_size,
            bbox2[1] : bbox2[1] + self.image_size
        ]

        return (
            image1, depth1, intrinsics1, pose1, bbox1,
            image2, depth2, intrinsics2, pose2, bbox2
        )

    def crop(self, image1, image2, central_match):
        bbox1_i = max(int(central_match[0]) - self.image_size // 2, 0)
        if bbox1_i + self.image_size >= image1.shape[0]:
            bbox1_i = image1.shape[0] - self.image_size
        bbox1_j = max(int(central_match[1]) - self.image_size // 2, 0)
        if bbox1_j + self.image_size >= image1.shape[1]:
            bbox1_j = image1.shape[1] - self.image_size

        bbox2_i = max(int(central_match[2]) - self.image_size // 2, 0)
        if bbox2_i + self.image_size >= image2.shape[0]:
            bbox2_i = image2.shape[0] - self.image_size
        bbox2_j = max(int(central_match[3]) - self.image_size // 2, 0)
        if bbox2_j + self.image_size >= image2.shape[1]:
            bbox2_j = image2.shape[1] - self.image_size

        return (
            image1[
                bbox1_i : bbox1_i + self.image_size,
                bbox1_j : bbox1_j + self.image_size
            ],
            np.array([bbox1_i, bbox1_j]),
            image2[
                bbox2_i : bbox2_i + self.image_size,
                bbox2_j : bbox2_j + self.image_size
            ],
            np.array([bbox2_i, bbox2_j])
        )
    
    def uv_to_pos(self, uv):
        return torch.cat([uv[1, :].view(1, -1), uv[0, :].view(1, -1)], dim=0)

    '''
    def interpolate_depth(self, pos, depth):
        device = pos.device

        ids = torch.arange(0, pos.size(1), device=device)

        i = pos[0, :].long()
        j = pos[1, :].long()

        # Valid depth
        valid_depth = torch.Tensor([depth[ii][jj] > 0 for ii, jj in zip(i, j)]).bool()

        ids = ids[valid_depth]
        if ids.size(0) == 0:
            raise EmptyTensorError

        i = i[ids]
        j = j[ids]
        interpolated_depth = torch.Tensor([depth[ii][jj] for ii, jj in zip(i, j)]).float()

        pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

        return [interpolated_depth, pos, ids]
    '''

    def interpolate_depth(self, pos, depth):
        device = pos.device

        ids = torch.arange(0, pos.size(1), device=device)

        h, w = depth.size()

        i = pos[0, :]
        j = pos[1, :]

        # Valid corners
        i_top_left = torch.floor(i).long()
        j_top_left = torch.floor(j).long()
        valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

        i_top_right = torch.floor(i).long()
        j_top_right = torch.ceil(j).long()
        valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

        i_bottom_left = torch.ceil(i).long()
        j_bottom_left = torch.floor(j).long()
        valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

        i_bottom_right = torch.ceil(i).long()
        j_bottom_right = torch.ceil(j).long()
        valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

        valid_corners = torch.min(
            torch.min(valid_top_left, valid_top_right),
            torch.min(valid_bottom_left, valid_bottom_right)
        )

        i_top_left = i_top_left[valid_corners]
        j_top_left = j_top_left[valid_corners]

        i_top_right = i_top_right[valid_corners]
        j_top_right = j_top_right[valid_corners]

        i_bottom_left = i_bottom_left[valid_corners]
        j_bottom_left = j_bottom_left[valid_corners]

        i_bottom_right = i_bottom_right[valid_corners]
        j_bottom_right = j_bottom_right[valid_corners]

        ids = ids[valid_corners]
        if ids.size(0) == 0:
            raise EmptyTensorError

        # Valid depth
        valid_depth = torch.min(
            torch.min(
                depth[i_top_left, j_top_left] > 0,
                depth[i_top_right, j_top_right] > 0
            ),
            torch.min(
                depth[i_bottom_left, j_bottom_left] > 0,
                depth[i_bottom_right, j_bottom_right] > 0
            )
        )

        i_top_left = i_top_left[valid_depth]
        j_top_left = j_top_left[valid_depth]

        i_top_right = i_top_right[valid_depth]
        j_top_right = j_top_right[valid_depth]

        i_bottom_left = i_bottom_left[valid_depth]
        j_bottom_left = j_bottom_left[valid_depth]

        i_bottom_right = i_bottom_right[valid_depth]
        j_bottom_right = j_bottom_right[valid_depth]

        ids = ids[valid_depth]
        if ids.size(0) == 0:
            raise EmptyTensorError

        # Interpolation
        i = i[ids]
        j = j[ids]
        dist_i_top_left = i - i_top_left.float()
        dist_j_top_left = j - j_top_left.float()
        w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
        w_top_right = (1 - dist_i_top_left) * dist_j_top_left
        w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
        w_bottom_right = dist_i_top_left * dist_j_top_left

        interpolated_depth = (
            w_top_left * depth[i_top_left, j_top_left] +
            w_top_right * depth[i_top_right, j_top_right] +
            w_bottom_left * depth[i_bottom_left, j_bottom_left] +
            w_bottom_right * depth[i_bottom_right, j_bottom_right]
        )

        pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

        return [interpolated_depth, pos, ids]
    
    def compute_all_matches(self, keypoints1, image1, depth1, intrinsics1, pose1, bbox1, keypoints2, image2, depth2, intrinsics2, pose2, bbox2):
        # keypoint: (1, -1, 2)
        # image: (3, 256, 256)
        # depth: (256, 256)
        # intrinsics: (3, 3)
        # pose: (4, 4)
        # bbox: (2)
        
        keypoints1 = keypoints1.reshape(-1, 2).T.astype(np.float32)
        keypoints1 = torch.from_numpy(keypoints1).cuda()
        image1 = torch.from_numpy(image1.astype(np.float32)).cuda()
        depth1 = torch.from_numpy(depth1.astype(np.float32)).cuda()
        intrinsics1 = torch.from_numpy(intrinsics1.astype(np.float32)).cuda()
        pose1 = torch.from_numpy(pose1.astype(np.float32)).cuda()
        bbox1 = torch.from_numpy(bbox1.astype(np.float32)).cuda()
        keypoints2 = keypoints2.reshape(-1, 2).T.astype(np.float32)
        keypoints2 = torch.from_numpy(keypoints2).cuda()
        image2 = torch.from_numpy(image2.astype(np.float32)).cuda()
        depth2 = torch.from_numpy(depth2.astype(np.float32)).cuda()
        intrinsics2 = torch.from_numpy(intrinsics2.astype(np.float32)).cuda()
        pose2 = torch.from_numpy(pose2.astype(np.float32)).cuda()
        bbox2 = torch.from_numpy(bbox2.astype(np.float32)).cuda()

        # print('keypoints shape:', keypoints1.shape, keypoints2.shape)

        # 图片系（2维坐标、深度）->相机系（3维）->世界系（3维）
        # 世界系里面找匹配

        device = keypoints1.device
        Z1, pos1, id1s = self.interpolate_depth(keypoints1, depth1)
        Z2, pos2, id2s = self.interpolate_depth(keypoints2, depth2)

        # COLMAP convention
        # bbox: 由于训练的图片是原始图片中截取的一部分，因此特征点在照片系里面真实的坐标需要加上裁剪偏置bbox
        u1 = pos1[1, :] + bbox1[1] + .5
        v1 = pos1[0, :] + bbox1[0] + .5
        u2 = pos2[1, :] + bbox2[1] + .5
        v2 = pos2[0, :] + bbox2[0] + .5

        X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
        Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])
        X2 = (u2 - intrinsics2[0, 2]) * (Z2 / intrinsics2[0, 0])
        Y2 = (v2 - intrinsics2[1, 2]) * (Z2 / intrinsics2[1, 1])

        XYZ1_hom = torch.cat([
            X1.view(1, -1),
            Y1.view(1, -1),
            Z1.view(1, -1),
            torch.ones(1, Z1.size(0), device=device)
        ], dim=0)
        XYZ1_hom = torch.matmul(torch.inverse(pose1), XYZ1_hom)
        XYZ1 = XYZ1_hom[:-1,:]
        # XYZ1_hom = torch.einsum('ab,bn->an',torch.inverse(pose1), XYZ1_hom)
        # XYZ1 = XYZ1_hom[: -1, :] / XYZ1_hom[-1, :].view(1, -1)
        XYZ2_hom = torch.cat([
            X2.view(1, -1),
            Y2.view(1, -1),
            Z2.view(1, -1),
            torch.ones(1, Z2.size(0), device=device)
        ], dim=0)
        XYZ2_hom = torch.matmul(torch.inverse(pose2), XYZ2_hom)
        XYZ2 = XYZ2_hom[:-1,:]
        # XYZ2 = XYZ2_hom[: -1, :] / XYZ2_hom[-1, :].view(1, -1)

        start_time = time.time()
        
        # 找匹配
        # ids：第一张图剩余点编号
        # pos1_to_2：第一张图剩余点在第二张图中的坐标
        # annotated_depth, estimated_depth：第一张图剩余点在第二张图中的深度（根据depth1和depth2计算值，根据depth1估计值）
        # id2s：第二张图剩余点编号
        # pos2：第二张图剩余点在第二张图中的坐标
        # Z2：第二张图剩余点在第二张图中的深度
        # print('ids, id2s shape:', ids.shape, id2s.shape)
        # print('ids, id2s:', ids, id2s)
        # print('pos1_to_2:', pos1_to_2)
        # print('pos2:', pos2)
        # print('annotated_depth:', annotated_depth)
        # print('estimated_depth:', estimated_depth)
        # print('Z2:', Z2)

        '''
        # 方法一：暴力实现
        matches_ids1 = []
        matches_ids2 = []
        flag1 = [1] * keypoints1.shape[1]
        flag2 = [1] * keypoints2.shape[1]
        # 寻找所有匹配
        # 每个点最多匹配一次
        for i1 in range(id1s.shape[0]):
            id1 = int(id1s[i1].item())
            for i2 in range(id2s.shape[0]):
                id2 = int(id2s[i2].item())
                if (flag2[id2] and self.distance_valid(XYZ1[:, i1], XYZ2[:, i2])):
                    matches_ids1.append(id1)
                    matches_ids2.append(id2)
                    flag1[id1] = 0
                    flag2[id2] = 0
                if (flag1[id1] == 0):
                    break

        if len(matches_ids1) == 0:
            raise EmptyTensorError
        
        # print('keypoints, matches shape:', keypoints1.shape, keypoints2.shape, len(matches_ids1))
        # 添加不存在的匹配（SuperGlue训练代码要求）
        for i in range(keypoints1.shape[1]):
            if flag1[i]:
                matches_ids1.append(i)
                matches_ids2.append(keypoints2.shape[1])
        for i in range(keypoints2.shape[1]):
            if flag2[i]:
                matches_ids1.append(keypoints1.shape[1])
                matches_ids2.append(i)
        '''
        
        '''
        # 方法二：jit加速（效果甚微）
        matches_ids1, matches_ids2 = ComputeMatches(keypoints1, keypoints2, id1s, id2s, XYZ1, XYZ2)
        '''

        # 方法三：调库
        matches_ids1 = []
        matches_ids2 = []
        flag1 = [True] * keypoints1.shape[1]
        flag2 = [True] * keypoints2.shape[1]
        # 寻找所有匹配

        sqr1 = torch.sum(XYZ1**2, dim=0).reshape(-1, 1)
        sqr2 = torch.sum(XYZ2**2, dim=0).reshape(1, -1)
        inner = torch.einsum('an,am->nm', XYZ1, XYZ2)
        D = -2 * inner + sqr1 + sqr2
        D = D ** 0.5
        M = D.argmin(dim=1)

        # D = cdist(XYZ1.T.cpu(), XYZ2.T.cpu(), metric='euclidean')
        # M = np.argmin(D, axis=1)
        for i in range(id1s.shape[0]):
            id1 = int(id1s[i].item())
            id2 = int(id2s[M[i]].item())
            if (flag1[id1] and flag2[id2] and (D[i][M[i]] < 0.02)):
                matches_ids1.append(id1)
                matches_ids2.append(id2)
                flag1[id1] = False
                flag2[id2] = False
        
        draw_matches(image1, keypoints1, image2, keypoints2, (matches_ids1, matches_ids2))
        exit()

        # print('keypoints, matches shape:', keypoints1.shape, keypoints2.shape, len(matches_ids1))
        # 添加不存在的匹配（SuperGlue训练代码要求）
        for i in range(keypoints1.shape[1]):
            if flag1[i]:
                matches_ids1.append(i)
                matches_ids2.append(keypoints2.shape[1])
        for i in range(keypoints2.shape[1]):
            if flag2[i]:
                matches_ids1.append(keypoints1.shape[1])
                matches_ids2.append(i)

        # print('matching time:', time.time() - start_time)

        # 少于5个匹配则跳过此样本
        if (keypoints1.shape[1] + keypoints2.shape[1] - len(matches_ids1)) < 5:
            raise EmptyTensorError

        matches_ids1 = np.array(matches_ids1)
        matches_ids2 = np.array(matches_ids2)
        all_matches = list(np.concatenate([matches_ids1.reshape(1, -1), matches_ids2.reshape(1, -1)], axis=0))
        # all_matches = np.concatenate([matches_ids1.reshape(1, -1), matches_ids2.reshape(1, -1)], axis=0)

        # print('keypoints, matches shape:', keypoints1.shape, keypoints2.shape, len(matches_ids1))

        # all_matches = list(np.array([[0,1],[0,1]]))
        return all_matches

    def __getitem__(self, idx):
        (
            image1, depth1, intrinsics1, pose1, bbox1,
            image2, depth2, intrinsics2, pose2, bbox2
        ) = self.recover_pair(self.dataset[idx])

        image1 = preprocess_image(image1, preprocessing=self.preprocessing) # 得到BGR格式图像，不做中心化
        image2 = preprocess_image(image2, preprocessing=self.preprocessing) # 得到BGR格式图像，不做中心化
        original_image1 = image1
        original_image2 = image2

        '''
        return {
            'image1': torch.from_numpy(image1.astype(np.float32)),
            'depth1': torch.from_numpy(depth1.astype(np.float32)),
            'intrinsics1': torch.from_numpy(intrinsics1.astype(np.float32)),
            'pose1': torch.from_numpy(pose1.astype(np.float32)),
            'bbox1': torch.from_numpy(bbox1.astype(np.float32)),
            'image2': torch.from_numpy(image2.astype(np.float32)),
            'depth2': torch.from_numpy(depth2.astype(np.float32)),
            'intrinsics2': torch.from_numpy(intrinsics2.astype(np.float32)),
            'pose2': torch.from_numpy(pose2.astype(np.float32)),
            'bbox2': torch.from_numpy(bbox2.astype(np.float32))
        }
        '''

        # 用SuperPoint处理两张图片，得到keypoints、descriptors、scores
        sift = self.sift
        image1 = np.transpose(image1, (1, 2, 0))
        image1 = Image.fromarray(np.uint8(image1))
        image1 = image1.convert('L')
        image1 = np.array(image1)
        image2 = np.transpose(image2, (1, 2, 0))
        image2 = Image.fromarray(np.uint8(image2))
        image2 = image2.convert('L')
        image2 = np.array(image2)
        kp1, descs1 = sift.detectAndCompute(image1, None)
        kp2, descs2 = sift.detectAndCompute(image2, None)

        # limit the number of keypoints
        kp1_num = min(self.nfeatures, len(kp1))
        kp2_num = min(self.nfeatures, len(kp2))
        kp1 = kp1[:kp1_num]
        kp2 = kp2[:kp2_num]

        kp1_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp1])
        kp2_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp2])

        # skip this image pair if no keypoints detected in image
        # 不到10个特征点也跳过此图片对
        if len(kp1) < 10 or len(kp2) < 10:
            return{
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': image1,
                'image1': image2,
                'file_name': ''
            } 

        # confidence of each key point
        scores1_np = np.array([kp.response for kp in kp1])
        scores2_np = np.array([kp.response for kp in kp2])

        kp1_np = kp1_np[:kp1_num, :]
        kp2_np = kp2_np[:kp2_num, :]
        descs1 = descs1[:kp1_num, :]
        descs2 = descs2[:kp2_num, :]

        kp1_np = kp1_np.reshape((1, -1, 2))
        kp2_np = kp2_np.reshape((1, -1, 2))
        descs1 = np.transpose(descs1 / 256.)
        descs2 = np.transpose(descs2 / 256.)

        image1 = torch.from_numpy(image1/255.).double()[None].cuda()
        image2 = torch.from_numpy(image2/255.).double()[None].cuda()

        # print(image1.shape, image2.shape, depth1.shape, depth2.shape)

        # 根据10元组和keypoints，得到所有匹配，按SuperGlue的输入要求返回结果
        # image1, depth1, intrinsics1, pose1, bbox1
        # image2, depth2, intrinsics2, pose2, bbox2
        # depth: (256, 256), intrinsics: (3, 3), pose: (4, 4), bbox: (2)
        # 例子：all_matches = list(np.array([[0],[0]]))
        try:
            all_matches = self.compute_all_matches(kp1_np, original_image1, depth1, intrinsics1, pose1, bbox1, kp2_np, original_image2, depth2, intrinsics2, pose2, bbox2)
        except EmptyTensorError:
            return {
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': image1,
                'image1': image2,
                'file_name': ''
            }

        # print(kp1_np.shape, kp2_np.shape, len(all_matches[0]))

        return {
            'keypoints0': list(kp1_np),
            'keypoints1': list(kp2_np),
            'descriptors0': list(descs1),
            'descriptors1': list(descs2),
            'scores0': list(scores1_np),
            'scores1': list(scores2_np),
            'image0': image1,
            'image1': image2,
            'all_matches': all_matches,
            'file_name': ''
        }

        # SuperGlue要的返回值
        '''
        return{
            'keypoints0': list(kp1_np),
            'keypoints1': list(kp2_np),
            'descriptors0': list(descs1),
            'descriptors1': list(descs2),
            'scores0': list(scores1_np),
            'scores1': list(scores2_np),
            'image0': image,
            'image1': warped,
            'all_matches': list(all_matches),
            'file_name': file_name
        } 
        '''
