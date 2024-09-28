import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
from torchvision.datasets import MNIST, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils, Flowers102
import PIL
import os
import os.path
import logging
import torch

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass



class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]  ## 根据索引选择数据
            target = target[self.dataidxs]  ## 根据索引选择数据

 
        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class MNIST_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        dataobj = MNIST(self.root, self.train, self.download, self.transform)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = dataobj.train_data, np.array(dataobj.train_labels)
            else:
                data, target = dataobj.test_data, np.array(dataobj.test_labels)
        else:
            data = dataobj.data
            target = np.array(dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]  ## 根据索引选择数据
            target = target[self.dataidxs]  ## 根据索引选择数据
 
        return data, target

    # def truncate_channel(self, index):
    #     for i in range(index.shape[0]):
    #         gs_index = index[i]
    #         self.data[gs_index, :, :, 1] = 0.0
    #         self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class. 
        """
        img, target = self.data[index].unsqueeze(0).float(), self.target[index]
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class EMNIST_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        dataobj = EMNIST(self.root, split='balanced', train=self.train, download=self.download, transform=self.transform)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = dataobj.train_data, np.array(dataobj.train_labels)
            else:
                data, target = dataobj.test_data, np.array(dataobj.test_labels)
        else:
            data = dataobj.data
            target = np.array(dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]  ## 根据索引选择数据
            target = target[self.dataidxs]  ## 根据索引选择数据
 
        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index].unsqueeze(0).float(), self.target[index]


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
        
class SVHN_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        dataobj = SVHN(self.root, split='train', download=self.download, transform=self.transform)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = dataobj.train_data, np.array(dataobj.train_labels)
            else:
                data, target = dataobj.test_data, np.array(dataobj.test_labels)
        else:
            data = dataobj.data
            target = np.array(dataobj.labels)

        if self.dataidxs is not None:
            data = data[self.dataidxs]  ## 根据索引选择数据
            target = target[self.dataidxs]  ## 根据索引选择数据
        
        data = data.reshape(data.shape[0], data.shape[2], data.shape[3], data.shape[1])

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class. .unsqueeze(0).float()
        """
        img, target = self.data[index], self.target[index]
    

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self): 
        return len(self.data)

class Flowers102FL(data.Dataset):
    def __init__(self,root,dataidxs=None,train=True,x_transform=None, y_transform=None, download=False, other_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.download = download
        self.x, self.y = self.__build_truncated_dataset__()
        self.other_transform=other_transform

    def __build_truncated_dataset__(self):

        if self.train:
            cifar_data_obj = Flowers102(self.root, 'test', self.x_transform, self.y_transform, self.download)
        else:
            cifar_data_obj = Flowers102(self.root, 'train', self.x_transform, self.y_transform, self.download)
        
        
        imgs = cifar_data_obj._image_files
        labels = np.array(cifar_data_obj._labels)

        if self.dataidxs is not None:
            imgslist=[]
            labelslist=[]
            for idx in self.dataidxs:
                imgslist.append(imgs[idx])
                labelslist.append(labels[idx])
            imgs, labels=imgslist, labelslist
        return imgs, labels
        

    def __getitem__(self, index):
        img_f, label = self.x[index], self.y[index]
        img_o = PIL.Image.open(img_f).convert("RGB")
        if self.x_transform is not None:
            img = self.x_transform(img_o)
        if self.y_transform is not None:
            label = self.y_transform(label)
        if self.other_transform == None:
            return img, label
        else:
            imgs=[]
            imgs.append(img)
            for t in self.other_transform:
                imgs.append(t(img_o))
            return imgs, label

    def __len__(self):
        return len(self.x)


class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)
