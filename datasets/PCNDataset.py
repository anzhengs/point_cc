import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import json
from .build import DATASETS
from utils.logger import *
from scipy.spatial.transform import Rotation as SciPyRot

def apply_random_rotation(partial_pc, gt_pc):
    """
    partial_pc: 输入的残缺点云, numpy array (N, 3)
    gt_pc: 真实的完整点云, numpy array (M, 3)
    """
    # 随机生成一个 3x3 的 SO(3) 旋转矩阵
    rot_matrix = SciPyRot.random().as_matrix().astype(np.float32)
    
    # 将相同的旋转矩阵应用到输入和 Ground Truth 上
    rotated_partial = np.dot(partial_pc, rot_matrix.T)
    rotated_gt = np.dot(gt_pc, rot_matrix.T)
    
    # 强制内存连续，防止后端 torch.bmm 报错
    return np.ascontiguousarray(rotated_partial), np.ascontiguousarray(rotated_gt)

@DATASETS.register_module()
class PCN(data.Dataset):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS
        self.random_rotation = getattr(config, 'RANDOM_ROTATION', False)

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            transforms_list = [
                {
                    'callback': 'RandomSamplePoints',
                    'parameters': {'n_points': 2048},
                    'objects': ['partial']
                }
            ]
            
            # 只在未使用随机旋转时才做镜像
            if not self.random_rotation:
                transforms_list.append({
                    'callback': 'RandomMirrorPoints',
                    'objects': ['partial', 'gt']
                })
                
            transforms_list.append({
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            })
            
            return data_transforms.Compose(transforms_list)
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
            samples = dc[subset]

            for s in samples:
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_path': [
                        self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path':
                    self.complete_points_path % (subset, dc['taxonomy_id'], s),
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            if isinstance(file_path, list):
                # 确保索引不越界，如果列表只有一个元素（如验证集），则取第0个
                real_idx = rand_idx if len(file_path) > rand_idx else 0
                file_path = file_path[real_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        assert data['gt'].shape[0] == self.npoints

        # 应用随机旋转
        # PCNdataset.py 中 _get_transforms：
        # train时的transforms包含RandomMirrorPoints
        # 而旋转在transforms之前应用

        if self.random_rotation:
            data['partial'], data['gt'] = apply_random_rotation(data['partial'], data['gt'])  # SO(3)

        if self.transforms is not None:
            data = self.transforms(data)  # 包含镜像！

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)