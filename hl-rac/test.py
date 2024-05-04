from dataset import Flag3d
import lightning as L
from models.lrac import LRAC
import torch.utils.data as data
from torch.utils.data import DataLoader,Sampler
import joblib
import random

class SubsetSampler(Sampler):
    def __init__(self,sub_index):
        random.shuffle(sub_index)
        self.sub_index = sub_index

    def __iter__(self):
        return iter(self.sub_index)

    def __len__(self):
        return len(self.sub_index)

num_frames = 128
backbone = "agcn"
batch_size = 32
split = "test_rac"
trans_num_layer = 3
mode = "spatial"
dropout = 0
lr=1e-5

root_path = "../processed_data"
dir_path = "./exp/test"

model = LRAC.load_from_checkpoint("./exp/test_rac.ckpt",map_location="cuda:0")
trainer = L.Trainer(default_root_dir=dir_path,logger=False,accelerator="gpu",devices=[0],)

test_dataloader = DataLoader(Flag3d(path = root_path, split = "test_rac",num_frames = num_frames),batch_size=32,pin_memory=True,shuffle=False, num_workers=32)
trainer.test(model,dataloaders=test_dataloader)
