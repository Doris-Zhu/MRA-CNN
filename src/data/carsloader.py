import scipy.io as scio
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import torch.utils.data as data
import random

class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert("RGB")
            #raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

#trainfilepath: the train annotations file path
#imgroot: all the images to use
#images in car_test don't have class labels
def carsdata(
        trainfilepath="../../data/StandfordCars/devkit/cars_train_annos.mat",
        imgroot="../../data/StandfordCars/train/cars_train/",
        test_rate=0.2
):
    data = scio.loadmat(trainfilepath)
    traindata = data['annotations'][0]
    images_path = []
    images_label = []
    for item in traindata:
        imgpath = os.path.join(imgroot, item[5][0])
        imglabel = item[4][0][0]
        images_path.append(imgpath)
        images_label.append(imglabel)


    train_images_path = []
    train_images_label = []
    test_images_path = []
    test_images_label = []

    #split
    imgnum=len(images_path)
    testimgindex=random.sample(range(imgnum), k=int(imgnum* test_rate))

    for i in range(imgnum):
        if i in testimgindex:
            test_images_path.append(images_path[i])
            test_images_label.append(images_label[i]-1)
        else:
            train_images_path.append(images_path[i])
            train_images_label.append(images_label[i]-1)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(448),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "test": transforms.Compose([transforms.Resize(448),
                                   transforms.CenterCrop(448),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    test_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform["test"])

    return train_dataset,test_dataset

def carclassname(imglabel,filepath="../../data/StandfordCars/devkit/cars_meta.mat"):
    data = scio.loadmat(filepath)
    thedata=data['class_names'][0]
    return thedata[imglabel-1][0]
    
    

if __name__ == '__main__':
    trainset,testset=carsdata()

    trainloader = data.DataLoader(trainset, batch_size=32,
                                  shuffle=False, collate_fn=trainset.collate_fn, num_workers=1)
    for img, cls in trainloader:
        print(' [*] train images:', img.size())
        print(' [*] train class:', cls.size())
        break

    testloader = data.DataLoader(testset, batch_size=32,
                                 shuffle=False, collate_fn=testset.collate_fn, num_workers=1)

    for img, cls in testloader:
        print(' [*] test images:', img.size())
        print(' [*] test class:', cls.size())
        break







