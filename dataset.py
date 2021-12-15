import torch
import torch.nn as nn
from torch.utils import data
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class TestSet(Dataset):
    def __init__(self,data_dir,transform=None):
        self.data_dir=data_dir
        self.data=os.listdir(data_dir)
        self.transform=transform

    def __getitem__(self, i):
        image=Image.open(self.data_dir+"/"+self.data[i]).convert("RGB")        
        if self.transform is not None:
            image=self.transform(image)
        return image,self.data[i]

    def __len__(self):
        return len(self.data)    

class ImagenetSet(Dataset):
    def __init__(self,data_dir,transform=None):
        self.data_dir=data_dir
        self.data=os.listdir(data_dir)
        self.label_file="dataset/imagenet_mapping.txt"
        self.info_file="dataset/val_solution.txt"
        self.transform=transform
        self.name2lab={}
        self.nametolabel()

    def nametolabel(self):
        file=open(self.label_file,"r")
        idx=0
        lines=file.readlines()
        for line in lines:
            name=line.split()[0]
            self.name2lab[name]=idx
            idx+=1
        file.close()

    def __getitem__(self, i):
        image=Image.open(self.data_dir+"/"+self.data[i]).convert("RGB")
        info_file=open(self.info_file,"r")
        info_lines=info_file.readlines()
        for line in info_lines:
            if self.data[i].split(".")[0]==line.split(",")[0]:
                info=(line.split(",")[1]).split()
                j=0
                bounding_boxes=[]
                while(j<len(info)):
                    label=self.name2lab[info[j]]
                    box=[label,int(info[j+1]),int(info[j+2]),int(info[j+3]),int(info[j+4])]
                    bounding_boxes.append(box)
                    j+=5
                break
        if self.transform is not None:
            image=self.transform(image)
        return image,bounding_boxes,self.data[i]

    def __len__(self):
        return len(self.data)