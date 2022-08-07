import os
import re
import shutil
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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

#root:the original downloaded files.
#data_path:ordered files path
def petsdata(root="../../data/oxford-iiit-pet/images",data_path='../../data/oxford-iiit-pet/pets'):
    pets_class = []
    val_rate=0.2

    if not os.path.exists(data_path):
        os.mkdir(data_path)
        
        

    for file in os.listdir(root):
        tmp = re.split('[_.]', file)
        tmp_class = ''
        for item in tmp:
            if item.isdigit():
                break
            else:
                if tmp_class == '':
                    tmp_class = item
                else:
                    tmp_class = tmp_class + ' ' + item
        if tmp_class in pets_class:
            old_name = root + '/' + file
            new_name = data_path + '/' + tmp_class + '/' + file
            shutil.copyfile(old_name, new_name)
            continue
        else:
            pets_class.append(tmp_class)
            if not os.path.exists(data_path + '/' + tmp_class):
                os.mkdir(data_path + '/' + tmp_class)
            old_name = root + '/' + file
            new_name = data_path + '/' + tmp_class + '/' + file
            shutil.copyfile(old_name, new_name)

    pets_class.sort()
    print(len(pets_class))
    print(pets_class)

    # index file
    class_indices = dict((k, v) for v, k in enumerate(pets_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('pets_class_indices.json', 'w') as json_file:
        json_file.write(json_str)
     
    
    root=data_path

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in pets_class:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla]

        every_class_num.append(len(images))

        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(448),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(448),
                                   transforms.CenterCrop(448),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    return train_dataset,val_dataset

if __name__ == '__main__':
    trainset,testset=petsdata()

    trainloader = data.DataLoader(trainset, batch_size=16,
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
