import torch
import os
import skvideo.io
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
import torch.nn  as nn
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision import models, transforms
import numpy as np
from scipy import spatial
import seaborn as sns
import os
import shutil
import matplotlib.pyplot as plt
import sys
import math
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BatchDataset(torch.utils.data.Dataset):

    def __init__(self, files, transform=None):
        
        self.files = files
        self.transform = transform
       
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pos_path=self.files[idx]
        neg_idx=np.random.choice(len(self.files))
        while neg_idx==idx:
            neg_idx=np.random.choice(len(self.files))
        neg_path=self.files[neg_idx]
        pos_videos=skvideo.io.vread(pos_path).transpose(0, 3, 1, 2)/255.0
        neg_videos=skvideo.io.vread(neg_path).transpose(0, 3, 1, 2)/255.0

        pos_idx=np.random.choice(len(pos_videos),2,replace=False)
        neg_idx=np.random.choice(len(neg_videos))
        

        sample = {"anchor":pos_videos[pos_idx[0]],"pos":pos_videos[pos_idx[1]],"neg":neg_videos[neg_idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample



