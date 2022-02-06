import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

def mycat():
    return 0

class CBIS_DDSM(Dataset):
    
    def __init__(self, data_path, train=True, size1=270, size=256):
        
        self.size1 = size1 # first resize size
        self.size = size  # crop size
        self.train = train
        
        self.img_path = []
        self.mask_path = []
        self.labels = []
        
        if self.train:
            
            df = pd.read_csv(os.path.join(data_path,"annotations_train.csv"))
            img_paths = df["image_full_png_path"].values
            mask_paths = df["image_mask_png_path"].values
            targets = df["binary pathology"].values
            
            self.img_path = [os.path.join(data_path,"train","full",x) for x in img_paths]
            self.mask_path = [os.path.join(data_path,"train","merged_masks",x) for x in mask_paths]
            self.labels = targets

        else:  
            #test set    
            df = pd.read_csv(os.path.join(data_path,"annotations_test.csv"))
            img_paths = df["image_full_png_path"].values
            mask_paths = df["image_mask_png_path"].values
            targets = df["binary pathology"].values
            
            self.img_path = [os.path.join(data_path,"test","full",x) for x in img_paths]
            self.mask_path = [os.path.join(data_path,"test","merged_masks",x) for x in mask_paths]
            self.labels = targets
            

    def transform_train(self, image, mask, size1=270, size=256):
        # Resize
        resize = transforms.Resize(size=(size1, size1))
        image = resize(image)
        mask = resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(size, size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        #mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
        
        # Normalize
        image = TF.normalize(image, mean=mean, std=std)
        # mask = TF.normalize(mask, mean=mean, std=std)              #do not normalize mask
        
        return image, mask
    
    def transform_test(self, image, mask, size1=270, size=256):
        # Resize
        resize = transforms.Resize(size=(size1, size1))
        image = resize(image)
        mask = resize(mask)

        # Random crop
        # i, j, h, w = transforms.CenterCrop.get_params(
        #     image, output_size=(size, size))
        image = TF.center_crop(image,size)
        mask = TF.center_crop(mask, size)

        # Random horizontal flipping
        # if random.random() > 0.5:
        #     image = TF.hflip(image)
        #     mask = TF.hflip(mask)

        # Random vertical flipping
        # if random.random() > 0.5:
        #     image = TF.vflip(image)
        #     mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
        #  Normalize
        image = TF.normalize(image, mean=mean, std=std)
        # mask = TF.normalize(mask, mean=mean, std=std)
        
        return image, mask
    
    def __getitem__(self, index):
        img, mask, target = self.img_path[index], self.mask_path[index], self.labels[index]
            
        img = cv2.imread(img)   #The image was saved using cv2. For some reason it only works if read using cv2 and then moving to PIL
        img  = Image.fromarray(img) # transform into PIL image
        # img = Image.open(img).convert('RGB')   
        mask = cv2.imread(mask)
        mask = Image.fromarray(mask).convert('1')
        # img = Image.open(img).convert('RGB')
        # mask = Image.open(mask).convert('RGB')
        
        # if self.transform is not None:
        #     img = self.transform(img)
        #     mask = self.transform(mask)    #need to garantee that the transform is the same for image and mask
        
        if self.train:
            img, mask = self.transform_train(img, mask,size1=self.size1, size=self.size)
        else:
            img, mask = self.transform_test(img, mask, size1=self.size1, size=self.size)
#         resize = transforms.Resize(size=(64, 64))
#         # img = resize(img)
#         img = TF.resize(img, (64, 64))
#         #mask = resize(mask)
#         mask = TF.resize(mask, (64, 64))
        
        
#         img = TF.to_tensor(img)
#         mask =  TF.to_tensor(mask)
            
        sample = {'image':img, 'mask': mask, 'target':target, 'index':index, 'path':self.img_path[index]}
        return sample


    def __len__(self):
        return len(self.img_path)
    
    
def get_loaders(args, **kwargs):
    
    trainset = CBIS_DDSM(data_path = args['data_path'], train=True, size1=args['size1'], size=args['size'])
    testset = CBIS_DDSM(data_path = args['data_path'], train=False, size1=args['size1'], size=args['size'])
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=4,
                pin_memory=True, **kwargs) #Normal training
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False, num_workers=4,
                pin_memory=True, **kwargs) #Normal test
    
    return train_loader, test_loader