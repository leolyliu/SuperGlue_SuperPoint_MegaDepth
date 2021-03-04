import h5py

import numpy as np
import os

import torch
from torch.utils.data import Dataset

import time

from tqdm import tqdm

from models.superpoint import SuperPoint
from models.utils import frame2tensor

from lib.utils import preprocess_image
from lib.exceptions import NoGradientError, EmptyTensorError

import cv2

def draw_matches(img1, cv_kpts1, img2, cv_kpts2, match_ids, match_color=(0, 255, 0), pt_color=(0, 0, 255)):
    print(f'matches:{len(match_ids[0])},{len(match_ids[1])}')
    cv_kpts1 = cv_kpts1.cpu().numpy().T
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
    cv2.imwrite('match_vis_tmp_' + str(time.time()) + '.png', display)
    return display
    
def get_depth_image(depth):        
    valid = depth>0
    depth_min,depth_max = depth[valid].min(),depth[valid].max()
    depth[valid] = (depth[valid]-depth_min) / (depth_max-depth_min) * 255
    depth = depth.astype(np.uint8)
    return depth

def interpolate_depth(pos, depth):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    h, w = depth.size()

    i = pos[1, :]
    j = pos[0, :]

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

class MegaDepthDataset(Dataset):
    def __init__(
            self,
            nfeatures=1024,
            scene_list_path='megadepth_utils/train_scenes.txt',
            scene_info_path='/local/dataset/megadepth/scene_info',
            base_path='/local/dataset/megadepth',
            train=True,
            preprocessing=None,
            min_overlap_ratio=0.1,
            max_overlap_ratio=0.7,
            max_scale_ratio=np.inf,
            pairs_per_scene=200,
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
        self.superpoint = SuperPoint({'max_keypoints': nfeatures}).cuda()

    def build_dataset(self):
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

            valid = np.logical_and(
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

    def __len__(self):
        return len(self.dataset)

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
        # image1 = Image.open(image_path1)
        # if image1.mode != 'RGB':
        #     image1 = image1.convert('RGB')
        # image1 = np.array(image1)
        image1 = cv2.imread(image_path1)
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
        # image2 = Image.open(image_path2)
        # if image2.mode != 'RGB':
        #     image2 = image2.convert('RGB')
        # image2 = np.array(image2)
        image2 = cv2.imread(image_path2)
        assert(image2.shape[0] == depth2.shape[0] and image2.shape[1] == depth2.shape[1])
        intrinsics2 = pair_metadata['intrinsics2']
        pose2 = pair_metadata['pose2']

        central_match = pair_metadata['central_match']
        image1, bbox1, image2, bbox2 = self.crop(image1, image2, central_match)

        # 不裁剪图片
        '''
        depth1 = depth1[
            bbox1[0] : bbox1[0] + self.image_size,
            bbox1[1] : bbox1[1] + self.image_size
        ]
        depth2 = depth2[
            bbox2[0] : bbox2[0] + self.image_size,
            bbox2[1] : bbox2[1] + self.image_size
        ]
        '''

        return (
            image1, depth1, intrinsics1, pose1, bbox1,
            image2, depth2, intrinsics2, pose2, bbox2
        )

    def crop(self, image1, image2, central_match):
        '''
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
        '''
        # 不裁剪图片
        return (
            image1,
            np.array([0, 0]),
            image2,
            np.array([0, 0])
        )
    
    def compute_all_matches(self, kp1, image1, depth1, intrinsics1, pose1, bbox1, kp2, image2, depth2, intrinsics2, pose2, bbox2):

        kp1 = torch.from_numpy(kp1.T.astype(np.double)).cuda()
        depth1 = torch.from_numpy(depth1.astype(np.double)).cuda()
        intrinsics1 = torch.from_numpy(intrinsics1.astype(np.double)).cuda()
        pose1 = torch.from_numpy(pose1.astype(np.double)).cuda()
        bbox1 = torch.from_numpy(bbox1.astype(np.double)).cuda()

        kp2 = torch.from_numpy(kp2.T.astype(np.double)).cuda()
        depth2 = torch.from_numpy(depth2.astype(np.double)).cuda()
        intrinsics2 = torch.from_numpy(intrinsics2.astype(np.double)).cuda()
        pose2 = torch.from_numpy(pose2.astype(np.double)).cuda()
        bbox2 = torch.from_numpy(bbox2.astype(np.double)).cuda()

        device = kp1.device

        # print(kp1.shape, depth1.shape, kp2.shape, depth2.shape)

        Z1, pos1, ids1 = interpolate_depth(kp1, depth1)
        Z2, pos2, ids2 = interpolate_depth(kp2, depth2)

        n1 = Z1.size(0)
        n2 = Z2.size(0)
        # print(kp1.shape[1], kp2.shape[1])
        # print(n1,n2)

        u1, v1 = pos1[1, :] + bbox1[1], pos1[0, :] + bbox1[0]
        u2, v2 = pos2[1, :] + bbox2[1], pos2[0, :] + bbox2[0]

        X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
        Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

        XYZ1_hom = torch.cat([
            X1.view(1, -1),
            Y1.view(1, -1),
            Z1.view(1, -1),
            torch.ones(1, n1, device=device)
        ], dim=0)
        XYZ2_hom = torch.chain_matmul(pose2, torch.inverse(pose1), XYZ1_hom)
        XYZ2 = XYZ2_hom[: -1, :] / XYZ2_hom[-1, :].view(1, -1)
        uv2_hom = torch.matmul(intrinsics2, XYZ2)

        Z2_map = XYZ2[-1, :]
        uv2_map = uv2_hom[: -1, :] / uv2_hom[-1, :].view(1, -1)

        uv2 = torch.cat([u2.view(1, -1),  v2.view(1, -1)], dim=0)

        matches_ids1 = []
        matches_ids2 = []
        flag1 = [True] * kp1.shape[1]
        flag2 = [True] * kp2.shape[1]

        sqr1 = torch.sum(uv2_map**2, dim=0).reshape(-1, 1)
        sqr2 = torch.sum(uv2**2, dim=0).reshape(1, -1)
        inner = torch.einsum('an,am->nm', uv2_map, uv2)
        D = -2 * inner + sqr1 + sqr2
        D = D ** 0.5
        M = D.argmin(dim=1)

        for i in range(n1):
            j = M[i]
            id1 = int(ids1[i].item())
            id2 = int(ids2[j].item())
            # print(i, M[i])
            depth_ratio = Z2_map[i] / Z2[j]
            depth_err = abs(Z2_map[i] - Z2[j])
            if flag1[id1] and flag2[id2] and D[i][j] <= 3 and depth_ratio>=0.9 and depth_ratio<=1.1 and depth_err<=0.05:
                matches_ids1.append(id1)
                matches_ids2.append(id2)
                flag1[id1] = False
                flag2[id2] = False
                # print(depth1[int(kp1[1][id1])][int(kp1[0][id1])],depth2[int(kp2[1][id2])][int(kp2[0][id2])])

        # draw_matches(image1, kp1, image2, kp2, (matches_ids1, matches_ids2))
        # depth1 = get_depth_image(depth1.cpu()[:,:, np.newaxis].repeat(1,1,3).numpy())
        # depth2 = get_depth_image(depth2.cpu()[:,:, np.newaxis].repeat(1,1,3).numpy())
        # draw_matches(depth1, kp1, depth2, kp2, (matches_ids1, matches_ids2))
        # print(bbox1, bbox2)
        # exit()

        for i in range(kp1.shape[1]):
            if flag1[i]:
                matches_ids1.append(i)
                matches_ids2.append(kp2.shape[1])
        for i in range(kp2.shape[1]):
            if flag2[i]:
                matches_ids1.append(kp1.shape[1])
                matches_ids2.append(i)

        # 少于5个匹配则跳过此样本
        if (kp1.shape[1] + kp2.shape[1] - len(matches_ids1)) < 5:
            raise EmptyTensorError

        matches_ids1 = np.array(matches_ids1)
        matches_ids2 = np.array(matches_ids2)
        all_matches = list(np.concatenate([matches_ids1.reshape(1, -1), matches_ids2.reshape(1, -1)], axis=0))
        return all_matches
    

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


    def __getitem__(self, idx):
        (
            image1, depth1, intrinsics1, pose1, bbox1,
            image2, depth2, intrinsics2, pose2, bbox2
        ) = self.recover_pair(self.dataset[idx])

        # SIFT
        # kp1, descs1 = self.sift.detectAndCompute(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), None)
        # kp2, descs2 = self.sift.detectAndCompute(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), None)

        # SuperPoint
        SuperPoint = self.superpoint
        kp1, descs1 = self.parse_superpoint_result(SuperPoint({'image': frame2tensor(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY))}))
        kp2, descs2 = self.parse_superpoint_result(SuperPoint({'image': frame2tensor(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY))}))

        # im_with_keypoints1 = cv2.drawKeypoints(image1, kp1, np.array([]), (255,0,0))
        # im_with_keypoints2 = cv2.drawKeypoints(image2, kp2, np.array([]), (255,0,0))
        # cv2.imwrite('match_000_kp1.png', im_with_keypoints1)
        # cv2.imwrite('match_000_kp2.png', im_with_keypoints2)

        kp1_num = min(self.nfeatures, len(kp1))
        kp2_num = min(self.nfeatures, len(kp2))

        if kp1_num < 10 or kp2_num < 10:
            return{
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': image1,
                'image1': image2,
                'file_name': ''
            } 
            
        kp1 = kp1[:kp1_num]
        kp2 = kp2[:kp2_num]
        descs1 = descs1[:kp1_num, :]
        descs2 = descs2[:kp2_num, :]

        # kp1_np = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1])
        # kp2_np = np.array([[kp.pt[0], kp.pt[1]] for kp in kp2])
        kp1_np = np.array([(kp[0], kp[1]) for kp in kp1])
        kp2_np = np.array([(kp[0], kp[1]) for kp in kp2])
        KP1 = kp1_np
        KP2 = kp2_np

        # scores1_np = np.array([kp.response for kp in kp1])
        # scores2_np = np.array([kp.response for kp in kp2])
        scores1_np = np.array([kp[2] for kp in kp1])
        scores2_np = np.array([kp[2] for kp in kp2])

        kp1_np = kp1_np[:kp1_num, :]
        kp2_np = kp2_np[:kp2_num, :]
        descs1 = descs1[:kp1_num, :]
        descs2 = descs2[:kp2_num, :]

        kp1_np = kp1_np.reshape((1, -1, 2))
        kp2_np = kp2_np.reshape((1, -1, 2))
        # descs1 = np.transpose(descs1 / 256.)
        # descs2 = np.transpose(descs2 / 256.)
        descs1 = np.transpose(descs1)
        descs2 = np.transpose(descs2)

        image1_o = image1
        image2_o = image2
        image1 = torch.from_numpy(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) / 255.).double()[None].cuda()
        image2 = torch.from_numpy(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) / 255.).double()[None].cuda()

        try:
            all_matches = self.compute_all_matches(KP1, image1_o, depth1, intrinsics1, pose1, bbox1, KP2, image2_o, depth2, intrinsics2, pose2, bbox2)
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
