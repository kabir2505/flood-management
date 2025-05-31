import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_train():
    train_tfms=transforms.Compose([
        transforms.Resize((150,150)), #resizes to 150x150
        
        #------data augmentation-------
        transforms.ColorJitter(brightness=(0.5,1.5)), #brightness range
        transforms.RandomRotation(30),
        transforms.RandomAffine(
            degrees=0, #since already rotated above
            shear=0.2, #shear range
            scale=(0.8,1.2) #zoom_range +-20%
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        
        transforms.ToTensor()
    
    ])

    train_dataset=datasets.ImageFolder("data/train",transform=train_tfms)
    print(train_dataset.class_to_idx)
    train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)


    return train_dataset,train_loader

