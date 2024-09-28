import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
import copy
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
import torchvision
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score

# from model import *
from datasets import CIFAR10_truncated, Flowers102FL, CIFAR100_truncated, ImageFolder_custom, MNIST_truncated, EMNIST_truncated, SVHN_truncated


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass



def load_cifar10_data(datadir):
    # transform = transforms.Compose([transforms.ToTensor()])
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(
            Variable(x.unsqueeze(0), requires_grad=False),
            (4, 4, 4, 4), mode='reflect').data.squeeze()),
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0),
        transforms.RandomCrop(32), ## 随机裁剪32*32
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target 
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_svhn_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])
    svhn_train_ds = SVHN_truncated(datadir, train=True, download=True, transform=transform)
    svhn_test_ds = SVHN_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'/train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'/val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)

def load_mnist_data():  
    train = torchvision.datasets.MNIST(root='./FL-KD/', train=True, download=True,
                                        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    X_train = train.data
    y_train = train.targets                             

    test = torchvision.datasets.MNIST(root='./FL-KD/', train=False, download=True,
                                        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    X_test = test.data
    y_test = test.targets

    return (X_train, y_train, X_test, y_test)

def load_emnist_data():
    train = torchvision.datasets.EMNIST(root='E:/FL_code/FL-KD/', split='balanced', train=True, download=True, 
                                        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    test = torchvision.datasets.EMNIST(root='E:/FL_code/FL-KD/', split='balanced', train=False, download=True, 
                                        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    
    X_train = train.data
    y_train = train.targets
    print(y_train.shape[0])

    X_test = test.data
    y_test = test.targets
    print(y_test.shape[0])
    return (X_train, y_train, X_test, y_test)

def load_flower102_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    data_train = Flowers102FL(datadir, train=True, download=True, x_transform=transform)
    data_test = Flowers102FL(datadir, train=False, download=True, x_transform=transform)

    x_train, y_train = data_train.x, data_train.y
    x_test, y_test = data_test.x, data_test.y
    
    return (x_train, y_train, x_test, y_test)

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    
    ## 记录每种类别的数据分别有多少
    for net_i, dataidx in net_dataidx_map.items():
        ## 返回第i个客户端分配的数据的种类与个数
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        ## 转为字典记录
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]

    ## 加上.item()，字典遍历方式, net_cls_counts是一个
    # for net_id, data in net_cls_counts.items():
    #     n_total=0
    #     for class_id, n_data in data.items():
    #         n_total += n_data
    #     data_list.append(n_total)
    # print('mean:', np.mean(data_list))
    # print('std:', np.std(data_list))
    # logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(dataset, datadir, partition, n_parties, beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)
    elif dataset =='mnist':
        X_train, y_train, X_test, y_test = load_mnist_data()
    elif dataset == 'emnist':
        X_train, y_train, X_test, y_test = load_emnist_data()
        # 若采用letters
    elif dataset == 'svhn':
        X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    elif dataset == 'flowers102':
        X_train, y_train, X_test, y_test = load_flower102_data(datadir)

    
    n_train = y_test.shape[0]  ## 训练数据的个数
    # print("******************")
    # print(n_train)
    # print("******************")
    # print(np.unique(y_train))

    if partition == "homo" or partition == "iid": 
        idxs = np.random.permutation(n_train) ## 生成打乱的索引，idxs为一个索引列表，a[idxs]为按照idxs索引输出
        batch_idxs = np.array_split(idxs, n_parties) 
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)} 
    
    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0 
        min_require_size = 10 
        K = np.unique(y_train).shape[0]
                                    
        N = y_train.shape[0]        
        net_dataidx_map = {}        

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)] ## 申请 n_parties个列表
            for k in range(K):
                idx_k = np.where(y_train == k)[0]  ## np.where返回符合要求数的坐标
                # 打乱坐标
                np.random.shuffle(idx_k)
                ## 采用迪利克雷分布，生成一个列表
                proportions = np.random.dirichlet(np.repeat(beta, n_parties)) 
                ## 
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                
                proportions = proportions / proportions.sum()
                
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return net_dataidx_map, traindata_cls_counts

class pack_data(Dataset):
    def __init__(self, data, truth):
        super(pack_data, self).__init__()
        self.data = data
        self.truth = truth
    
    def __len__(self):
        return self.truth.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.truth[index]

def get_trainable_parameters(net, device='cpu'):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    # print("net.parameter.data:", list(net.parameters()))
    paramlist = list(trainable)
    #print("paramlist:", paramlist)
    N = 0
    for params in paramlist:
        N += params.numel()
        # print("params.data:", params.data)
    X = torch.empty(N, dtype=torch.float64, device=device)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data))
        offset += numel
    # print("get trainable x:", X)
    return X


def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(loader):
                    #print("x:",x)
                    #print("target:",target)
                    if device != 'cpu':
                        x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                    
                    _, _, out = model(x)
                    if len(target)==1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    ## 计算标签
                    _, pred_label = torch.max(out.data, 1)
                    
                    loss_collector.append(loss.item())   
                    total += x.data.size()[0]
                    ## 此处去掉.item亦可
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        ## 将pred_label加入到pred_labels_list列表中
                        pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                        true_labels_list = np.append(true_labels_list, target.data.numpy())
                    else:
                        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            inter_feature = None
            for batch_idx, (x, target) in enumerate(dataloader):
                #print("x:",x)
                if device != 'cpu':
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                # target = target-1
                out, feature = model(x, return_feature=True)
                if inter_feature is None:
                    inter_feature = out
                else:
                    inter_feature = torch.cat([inter_feature,out],dim=0)
                
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
                
                ## 计算f-score, precision, recall


                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
            avg_loss = sum(loss_collector) / len(loss_collector)
            precision, recall, fscore, _ = precision_recall_fscore_support(true_labels_list,pred_labels_list,average='macro')

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    ## conf_matrix, inter_feature, avg_loss
    if get_confusion_matrix:
        return correct / float(total), precision, recall, fscore

    return correct / float(total), precision, recall, fscore

def compute_loss(model, dataloader, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True
    if device == 'cpu':
        criterion = nn.CrossEntropyLoss()
    elif "cuda" in device.type:
        criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            _,_,out = model(x)
            loss = criterion(out, target)
            loss_collector.append(loss.item())

        avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return avg_loss



def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir + "trained_local_model" + str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    #
    with open("trained_local_model" + str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    if device == "cpu":
        model.to(device)
    else:
        model.cuda()
    return model

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return x, self.transform(x)

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0):
    if dataset in ('cifar10', 'cifar100','svhn'):
        if dataset == 'cifar10':
            dl_obj = CIFAR10_truncated  

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                # transforms.Lambda(lambda x: F.pad(
                #     Variable(x.unsqueeze(0), requires_grad=False),
                #     (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                normalize])

        elif dataset == 'cifar100':
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

            transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # data prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                normalize])
        #SVHN
        elif dataset == 'svhn':
            dl_obj = SVHN_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                torchvision.transforms.Normalize((0.485,0.456,0.106), (0.229,0.224,0.225))])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                torchvision.transforms.Normalize((0.485,0.456,0.106), (0.229,0.224,0.225))])
                
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=0)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=0)
    
    elif dataset == 'mnist':
        transform_train = transforms.Compose([
                                        torchvision.transforms.Normalize((0.5,),(0.5,))])
        train_ds = MNIST_truncated(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)  

        test_ds = MNIST_truncated(datadir, train=False, transform=transform_train, download=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=0)
    
    elif dataset == 'emnist':
        transform_train = transforms.Compose([
                                        transforms.Resize((28,28)),
                                        # transforms.RandomHorizontalFlip(),
                                        # transforms.RandomCrop(size=28,
                                        # padding=int(28 * 0.125),
                                        # padding_mode='reflect'),  
                                        torchvision.transforms.Normalize((0.5,),(0.5,))])
        transform_test = transforms.Compose([
                                        transforms.Resize((28,28)),
                                        torchvision.transforms.Normalize((0.5,),(0.5,))])
 
        train_ds = EMNIST_truncated(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)  

        test_ds = EMNIST_truncated(datadir, train=False, transform=transform_test, download=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=0)
    
    elif dataset == 'tinyimagenet':
        dl_obj = ImageFolder_custom
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir+'/train/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir+'/val/', transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    elif dataset=='flowers102':

        normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                            std=[0.229,0.224,0.225])

        transform_train = transforms.Compose([
            transforms.RandomRotation(45), # 随机旋转 -45度到45度之间
            transforms.CenterCrop(224), # 从中心处开始裁剪
            # 以某个随机的概率决定是否翻转 55开
            transforms.RandomHorizontalFlip(p = 0.5), # 随机水平翻转
            transforms.RandomVerticalFlip(p = 0.5), # 随机垂直翻转
            # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            transforms.ColorJitter(brightness = 0.2, contrast = 0.1, saturation = 0.1, hue = 0.1),
            transforms.RandomGrayscale(p = 0.025), # 概率转换为灰度图，三通道RGB
            # 灰度图转换以后也是三个通道，但是只是RGB是一样的
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
        
        train_ds = Flowers102FL(datadir, dataidxs=dataidxs, train=True, download=True, x_transform=transform_train)
        test_ds = Flowers102FL(datadir, train=False, download=True, x_transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

                
    return train_dl, test_dl, train_ds, test_ds

