""" Dataset loader for the flag3d dataset """
import os
import json

import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.data import DataLoader
import math
from scipy import integrate


def PDF(x, u, sig):
    # f(x)
    return np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)

# integral f(x)
def get_integrate(x_1, x_2, avg, sig):
    y, err = integrate.quad(PDF, x_1, x_2, args=(avg, sig))
    return y

def root_algin_np(keypoints):
    root = np.expand_dims(keypoints[:,0,:],1)
    keypoints = keypoints - root
    return keypoints

def left2right(kp):
    kp[:,:,2]=-1*kp[:,:,2]
    return kp

class Flag3d(data.Dataset):

    def __init__(self,path,split,num_frames=256,):
        super(Flag3d, self).__init__()
        self.scale = 10
        data_root = os.path.join(path,split)
        skeleton_root = os.path.join(path,"skeleton")
        self.kpts = []
        for _, _, files in os.walk(data_root, topdown=False):
            pass
        files = sorted(files)
        files = self.remove_file_suffixes(files)
        self.split = split
        self.data = files
        self.data_root = data_root
        self.skeleton_root = skeleton_root
        self.toss = np.load(os.path.join(path,"aug_rac/{}_toss.npy".format(self.split)))
        self.part = np.load(os.path.join(path,"aug_rac/{}_part.npy".format(self.split)))
        self.num_frames = num_frames
        self.text_embedding = os.path.join(path,"aug_rac/text_embedding_rac.pt")
        self.embedding_table = torch.load(self.text_embedding)

    def __getitem__(self, index):
        name = self.data[index%len(self.data)]
        kpts = np.load(os.path.join(self.skeleton_root,name)+".npy")
        kpts = left2right(root_algin_np(kpts))
        kpts = torch.from_numpy(kpts)
        table = pd.read_csv(os.path.join(self.data_root,name)+".csv", sep=',',header=0)
        start_idx = np.array(table['Sta'])
        end_idx = np.array(table['End'])
        F,V,C = kpts.shape
        duration = kpts.shape[0]
        counts = len(start_idx)
        toss = self.toss[index]
        if toss <=0.25:
            pass
        else:
            part_point = self.part[index]
            if toss <= 0.625:
                try:
                    duration = end_idx[part_point]
                except:
                    pass
                kpts = kpts[:end_idx[part_point]]
                start_idx = start_idx[:part_point+1]
                end_idx = end_idx[:part_point+1]
            else:
                duration = duration-end_idx[part_point]
                benchmark = end_idx[part_point]
                kpts = kpts[end_idx[part_point]:]
                start_idx = start_idx[part_point+1:]
                start_idx = [i-benchmark for i in start_idx]
                end_idx = end_idx[part_point+1:]
                end_idx = [i -benchmark for i in end_idx]
        kpts = self.adjust_frames(kpts)
        kpts = torch.stack(kpts)
        time_points = [element for pair in zip(start_idx, end_idx) for element in pair]
        counting_label = preprocess(duration, time_points, num_frames=self.num_frames)
        counting_label = torch.tensor(counting_label).to(torch.float32)
        textual_input = self.embedding_table[name[0:4]]
        return kpts,textual_input,counting_label

    def __len__(self):
        return len(self.data*self.scale)

    def remove_file_suffixes(self,file_list):
        new_list = []
        for file_name in file_list:
            file_name_without_suffix = file_name.rsplit('.', 1)[0]
            new_list.append(file_name_without_suffix)
        return new_list
    
    def get_test_toss(self):
        toss=[]
        np.random.seed(3407)
        for i in range(len(self.data)*self.scale):
            toss.append(np.random.rand())
        return toss

    def adjust_frames(self, frames):
        """
        # adjust the number of total video frames to the target frame num.
        :param frames: original frames
        :return: target number of frames
        """
        frames_adjust = []
        frame_length = len(frames)
        if self.num_frames <= len(frames):
            for i in range(1, self.num_frames + 1):
                frame = frames[i * frame_length // self.num_frames - 1]
                frames_adjust.append(frame)
        else:
            for i in range(frame_length):
                frame = frames[i]
                frames_adjust.append(frame)
            for _ in range(self.num_frames - frame_length):
                if len(frames) > 0:
                    frame = frames[-1]
                    frames_adjust.append(frame)
        return frames_adjust  
    
def preprocess(video_frame_length, time_points, num_frames):
    """
    process label(.csv) to density map label
    Args:
        video_frame_length: video total frame number, i.e 1024frames
        time_points: label point example [1, 23, 23, 40,45,70,.....] or [0]
        num_frames: 64
    Returns: for example [0.1,0.8,0.1, .....]
    """
    new_crop = []
    for i in range(len(time_points)):  # frame_length -> 64
        item = min(math.ceil((float((time_points[i])) / float(video_frame_length)) * num_frames), num_frames - 1)
        new_crop.append(item)
    new_crop = np.sort(new_crop)
    label = normalize_label(new_crop, num_frames)
    return label

def normalize_label(y_frame, y_length):

    # y_length: total frames
    # return: normalize_label  size:nparray(y_length,)
    y_label = [0 for i in range(y_length)]  # 坐标轴长度，即帧数
    for i in range(0, len(y_frame), 2):
        x_a = y_frame[i]
        x_b = y_frame[i + 1]
        avg = (x_b + x_a) / 2
        sig = (x_b - x_a) / 6
        num = x_b - x_a + 1  # 帧数量 update 1104
        if num != 1:
            for j in range(num):
                x_1 = x_a - 0.5 + j
                x_2 = x_a + 0.5 + j
                y_ing = get_integrate(x_1, x_2, avg, sig)
                y_label[x_a + j] = y_ing
        else:
            y_label[x_a] = 1
    return y_label

def get_integrate(x_1, x_2, avg, sig):

    y, err = integrate.quad(PDF, x_1, x_2, args=(avg, sig))
    return y

def read_file(file_path):
  
  with open(file_path, "r") as infile:
    content = infile.read()
  return content

if __name__=='__main__':
    dataset = Flag3d("../processed_data","test_rac")
    dataloader = DataLoader(
                        dataset,
                        batch_size=64,
                        shuffle=False,
                        num_workers=16,
                        pin_memory=True,
                        )
    for i in tqdm(dataloader):
        pass
