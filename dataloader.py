from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import pandas as pd


class ImgDataset(Dataset):
    def __init__(self,img_size,img_path,pkl_path):
        Trans = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            # TODO: Test whether RandomCrop or CenterCrop or don't keep ratio is better
            # transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
        self.transform = Trans
        self.img_path = img_path
        with open(pkl_path,"rb") as f:
            self.img_id2attr = pickle.loads(f.read())
#        self.imgids = os.listdir(img_path)
        self.imgids = list(self.img_id2attr.keys())

    def __getitem__(self, index):
        img_id = self.imgids[index % len(self.imgids)]
        for n in ['.jpg','.png','.jpeg','.gif']:
            img_path = os.path.join(self.img_path, str(img_id)+n)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img_tensor = self.transform(img)
                img_attr = torch.zeros(512)
                for a in self.img_id2attr[img_id]:
                    img_attr[a] = 1
                return {'img': img_tensor, 'attr': img_attr}
        '''
        img_id = self.imgids[index % len(self.imgids)]
        img_path = os.path.join(self.img_path, img_id)
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        img_attr = torch.zeros(512)
        img_id = int(img_id.split(".")[0])
        for a in self.img_id2attr[img_id]:
            img_attr[a] = 1
        return {'img': img_tensor, 'attr': img_attr}
        '''

    def __len__(self):
        return len(self.imgids)


class ImgIter:
    def __init__(self, img_path, pkl_path, batch_size,img_size=256, n_workers=16):
        self.batch_size = batch_size
        dataset = ImgDataset(img_path=img_path,pkl_path=pkl_path,img_size=img_size)
        self.train_size = int(0.9 * dataset.__len__())
        self.test_size = dataset.__len__() - self.train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [self.train_size, self.test_size])
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=n_workers,
                                           drop_last=True,
                                           pin_memory=True)
        self.train_iter = iter(self.train_dataloader)
        self.test_dataloader = DataLoader(test_dataset,
                                           batch_size=int(batch_size),
                                           shuffle=True,
                                           num_workers=n_workers,
                                           drop_last=True,
                                           pin_memory=True)
        self.test_iter = iter(self.test_dataloader)

    def get_train_n_iters(self):
        return int(self.train_size / self.batch_size)

    def get_test_n_iters(self):
        return int(self.test_size / self.batch_size)

    def get_train_batch(self):
        try:
            batch = self.train_iter.__next__()
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            batch = self.train_iter.__next__()
        return batch

    def get_test_batch(self):
        try:
            batch = self.test_iter.__next__()
        except StopIteration:
            self.test_iter = iter(self.test_dataloader)
            batch = self.test_iter.__next__()
        return batch


if __name__ == "__main__":
    test = ImgIter("../test_imgs","./imgid2attr.pkl",batch_size=32)
    for i in range(100):
        print(i)
        b=test.get_train_batch()
        c=test.get_test_batch()
    print(b)
