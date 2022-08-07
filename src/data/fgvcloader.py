import scipy.io as scio
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import torch.utils.data as data
import json

#numclass=100

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


def fgvcdata(filepath="../../data/fgvc-aircraft-2013b/data/"):
    classes=[]
    with open(os.path.join(filepath,'variants.txt')) as f:
        for line in f.readlines():
            classes.append(line[:-1])
    print(len(classes))
    print(classes)
    # index file
    class_indices = dict((k, v) for v, k in enumerate(classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('fgvc_class_indices.json', 'w') as json_file:
        json_file.write(json_str)
            
    train_images_path = []
    train_images_label = []
    test_images_path = []
    test_images_label = []

    with open(os.path.join(filepath,'images_variant_train.txt')) as f:
        for line in f.readlines():
            tmp=line.split(" ", 1)
            train_images_label.append(classes.index(tmp[1][:-1]))
            train_images_path.append(os.path.join(filepath,f'images/{tmp[0]}.jpg'))

    with open(os.path.join(filepath,'images_variant_test.txt')) as f:
        for line in f.readlines():
            tmp=line.split(" ", 1)
            test_images_label.append(classes.index(tmp[1][:-1]))
            test_images_path.append(os.path.join(filepath,f'images/{tmp[0]}.jpg'))


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

if __name__ == '__main__':
    trainset,testset=fgvcdata()

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







