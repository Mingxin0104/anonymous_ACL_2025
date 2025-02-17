# coding: utf-8
# 导入所需模块
from C3D_model import *
import json
import torchvision
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import os
from torch import save, load
import pickle
import time
import numpy as np
import PIL.Image as Image
import collections
import skimage.io as io
from skimage.transform import resize
import h5py
import fnmatch
from PIL import Image


def feature_extractor():
    # 初始化参数
    # trainloader = Train_Data_Loader(VIDEO_DIR, resize_w=128, resize_h=171, crop_w=112, crop_h=112, nb_frames=16)
    net = C3D(487)
    print('net', net)
    # 加载预训练模型并调整最后一层
    net.load_state_dict(torch.load(''))
    if RUN_GPU:
        net.cuda(0)
    net.eval()
    print('net', net)
    feature_dim = 4096 if EXTRACTED_LAYER!= 5 else 8192
    video_list = os.listdir(VIDEO_DIR)
    print('video_list', video_list)
    # 创建输出目录如果不存在
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    output_file = h5py.File(os.path.join(OUTPUT_DIR, OUTPUT_NAME), 'w')

    # 统计特定前缀文件数量的函数
    def count_files(directory, prefix_list):
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x + '*')) for x in prefix_list]
        return cnt_list

    # 遍历视频列表提取特征
    for video_name in video_list:
        video_path = os.path.join(VIDEO_DIR, video_name)
        print('video_path', video_path)
        all_cnt = count_files(video_path, ('image_'))
        total_frames = all_cnt[0]
        print('Total frames: %d' % total_frames)
        valid_frames = total_frames // nb_frames * nb_frames
        print('Total validated frames: %d' % valid_frames)
        index_w = np.random.randint(resize_w - crop_w)
        index_h = np.random.randint(resize_h - crop_h)
        features = []
        print('NB features: %d' % (valid_frames // nb_frames))
        for i in range(valid_frames // nb_frames):
            clip = np.array([resize(io.imread(os.path.join(video_path, 'image_{:04d}.jpg'.format(j))),
                                   output_shape=(resize_w, resize_h), preserve_range=True)
                             for j in range(i * nb_frames + 1, (i + 1) * nb_frames + 1)])
            clip = clip[:, index_w: index_w + crop_w, index_h: index_h + crop_h, :]
            clip = torch.from_numpy(np.float32(clip.transpose(3, 0, 1, 2)))
            clip = Variable(clip).cuda() if RUN_GPU else Variable(clip)
            clip = clip.resize(1, 3, nb_frames, crop_w, crop_h)
            _, clip_output = net(clip, EXTRACTED_LAYER)
            clip_feature = (clip_output.data).cpu()
            features.append(clip_feature)
        features = torch.cat(features, 0)
        features = features.numpy()
        print('features', features)
        # 在 HDF5 文件中创建视频组并保存特征和帧数信息
        video_group = output_file.create_group(video_name)
        video_group.create_dataset('c3d_features', data=features)
        video_group.create_dataset('total_frames', data=np.array(total_frames))
        video_group.create_dataset('valid_frames', data=np.array(valid_frames))
        print('%s has been processed...' % video_name)
    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print('******--------- Extract C3D features ------*******')
    parser.add_argument('-o', '--OUTPUT_DIR', dest='OUTPUT_DIR', type=str, default='./output_frm/',
                        help='Output file name')
    parser.add_argument('-l', '--EXTRACTED_LAYER', dest='EXTRACTED_LAYER', type=int, choices=[5, 6, 7], default=5,
                        help='Feature extractor layer')
    parser.add_argument('-i', '--VIDEO_DIR', dest='VIDEO_DIR', type=str, help='Input Video directory')
    parser.add_argument('-gpu', '--gpu', dest='GPU', action='store_true', help='Run GPU?')
    parser.add_argument('--OUTPUT_NAME', default='c3d_features.hdf5', help='The output name of the hdf5 features')
    args = parser.parse_args()
    params = vars(args)
    print('parsed parameters:')
    print(json.dumps(params, indent=2))
    OUTPUT_DIR = params['c3d_feature']
    EXTRACTED_LAYER = params['6']
    VIDEO_DIR = params['/sda/tmp/videos/new']
    RUN_GPU = params['GPU']
    OUTPUT_NAME = params['OUTPUT_NAME']
    crop_w = 112
    resize_w = 128
    crop_h = 112
    resize_h = 171
    nb_frames = 16
    feature_extractor()