import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse 
import logging
import os
import copy
import datetime
import random
import scipy
from tqdm import tqdm 
import scipy.io as io
from para_setting import get_args
from resnet import *
from utils import *
from TRPCA import *

seed = 2022
scipy.random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_global_model(args):
    global_model = ResNet(args.model, args)
    global_model.cuda()
    return global_model

def get_local_models(args):
    local_models = [] 
    if args.model == "imagenet_sup_rn50" or "imagenet_sup_rn34" or "imagenet_sup_rn18":
        for i in range(args.n_clients):
            model = ResNet(args.model, args)
            model.cuda()
            local_models.append(model)
    if args.model == "mixed":
        for i in range(3):
            model = ResNet("imagenet_sup_rn50", args)
            model.cuda()
            local_models.append(model)
        for i in range(7):
            model = ResNet("imagenet_sup_rn34", args)
            model.cuda()
            local_models.append(model)
        for i in range(10):
            model = ResNet("imagenet_sup_rn18", args)
            model.cuda()
            local_models.append(model)
    return local_models

def train_model(args, model_id, model, train_dataloader, test_dataloader=None):
    ## 初始化优化器
    optimizer = optim.SGD(model.parameters(), args.adam_lr, weight_decay=args.weight_decay)
    ## 初始化交叉熵损失
    criterion = nn.CrossEntropyLoss().cuda()
    for epoch in range(args.epochs):
        loss_list = []
        for batch_index, (X, Y) in tqdm(enumerate(train_dataloader)):

            # X = torch.cat([X[0], X[1]], dim=0)
            X = X.cuda()
            Y = Y.cuda()
            Y = Y.long()

            print("******************")
            print(X.shape)
            print("******************")

            out = model(X)
            loss = criterion(out, Y)
            ## 清空优化器
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss)
        if len(loss_list)>0:
            loss_epoch = sum(loss_list)/len(loss_list)
        else:
            loss_epoch = sum(loss_list)
        logger.info("The {}-th model's {}-th epoch's loss is {}".format(model_id+1, epoch+1, loss_epoch))
    
    # test_acc, _ = compute_accuracy(model, test_dataloader, device="cuda")
    # logger.info("The {}-th model's test acc is {}".format(model_id+1, test_acc))
    

if __name__=='__main__':
    args = get_args()
    if os.path.exists(args.log_directory):
        pass
    else:
        os.makedirs(args.log_directory) 

    ## 定义日志文件的名称
    log_name = 'experiments_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.log'
    ## 指定日志文件的位置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(args.log_directory, log_name),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("FedGPT-OR running, model:{}, dataset:{}, prompt:{}, Dirichlet beta:{}, lr:{}, svd ratio:{}, first_round:{}".format(args.model, 
                                                                                          args.dataset, 
                                                                                          args.model_prompt_num_tokens, 
                                                                                          args.beta,
                                                                                          args.adam_lr,
                                                                                          args.ratio,
                                                                                          args.first_round))

    ## 获取全局，本地模型
    global_model = get_global_model(args)
    # for key in global_model.state_dict().keys():
    #     print(key)
    # print("total para:", end='')
    # print(sum(p.numel() for p in global_model.parameters()))
    # for name, para in global_model.named_parameters():
    #     print("layer name:", name)
    #     print('para count:', para.numel())


    local_models = get_local_models(args)

    ## 划分数据集
    eachClient_data_index, _ = partition_data(args.dataset, 
                                              args.data_directory,
                                              args.partition_style, 
                                              args.n_clients, 
                                              beta=args.beta)

    ## 开始训练
    for round in tqdm(range(args.commucation_times)):
        ## 随机选择客户端
        selected_clients = random.sample(range(0, args.n_clients), args.n_selected_clients)
        selected_clients = sorted(selected_clients)
        sample_sum = 0

        for i_client in selected_clients:            
            # 加载dataloader
            train_loader, test_loader, _ ,_  = get_dataloader(args.dataset, 
                                                              args.data_directory, 
                                                              args.batch_size,
                                                              args.batch_size,
                                                              eachClient_data_index[i_client])
                                        
            train_model(args, i_client, local_models[i_client], train_loader, test_loader)

            # 统计sample_sum
            sample_sum += len(eachClient_data_index[i_client])

            logger.info("The {}-th model completes training".format(i_client+1))
            logger.info("********************************")
        
        # 对conv1与mlp进行average, 对prompt进行低秩张量处理
        global_para = {}
        prompt_list = []
        client_weight_list = []
        for i_client in selected_clients:
            client_weight = len(eachClient_data_index[i_client])/sample_sum
            client_weight_list.append(client_weight)
            if len(global_para) == 0:
                for k in local_models[i_client].state_dict().keys():
                    if 'prompt_layers' in k or 'head' in k:
                        global_para[k] =  client_weight * local_models[i_client].state_dict()[k]
                    if 'prompt_embeddings' in k:
                        for i_prompt in range(args.model_prompt_num_tokens):
                            prompt_list.append(local_models[i_client].state_dict()[k][:,i_prompt,:,:])
                             
            else:
                for k in local_models[i_client].state_dict().keys():
                    if 'prompt_layers' in k or 'head' in k:
                        global_para[k] += client_weight *  local_models[i_client].state_dict()[k]
                    if 'prompt_embeddings' in k:
                        for i_prompt in range(args.model_prompt_num_tokens):
                            prompt_list[i_prompt] = torch.cat((prompt_list[i_prompt], \
                            local_models[i_client].state_dict()[k][:,i_prompt,:,:]), dim=0)
        
        # print("***************")
        # print(prompt_list[i_prompt].shape)
        # print("***************")        
        ## prompt低秩张量优化
        # global_prompt = torch.Tensor([1, args.model_prompt_num_tokens, args.data_cropsize, args.data_cropsize])
        global_prompt = None
        # 遍历prompt_list，长度为prompt的个数
        for i_prompt in range(args.model_prompt_num_tokens):
            i_prompt_new = None
            if i_prompt == 0:
                ## prompt_list[i_prompt]的shape为[10,224,224], 调整为[224,224,10]
                if (round+1)<=args.first_round:
                    prompt_tensor = T_SVD(round+1, prompt_list[i_prompt].view(args.data_cropsize, args.data_cropsize, args.n_selected_clients), k=int(224*args.ratio))
                    prompt_tensor = prompt_tensor.view(args.n_selected_clients, args.data_cropsize, args.data_cropsize)
                else:
                    prompt_tensor = prompt_list[i_prompt]
                # prompt_tensor, _ = ADMM(logger, prompt_list[i_prompt].cpu().numpy()) 
                # 遍历prompt_list中每个张量中的每一层，张量维度为selected_clients*N*N
                for i_client in range(len(selected_clients)):
                    if i_client == 0:
                        i_prompt_new = client_weight_list[i_client] * prompt_tensor[i_client,:,:]
                    else:
                        i_prompt_new = i_prompt_new + client_weight_list[i_client] * prompt_tensor[i_client,:,:]
                # i_prompt_new /= len(selected_clients)
                global_prompt = i_prompt_new.unsqueeze(0) 
                # global_prompt = torch.from_numpy(i_prompt_new).unsqueeze(0) 
            else:
                if (round+1)<=args.first_round:
                    prompt_tensor = T_SVD(round+1, prompt_list[i_prompt].view(args.data_cropsize, args.data_cropsize, args.n_selected_clients),k=int(224*args.ratio))
                    prompt_tensor = prompt_tensor.view(args.n_selected_clients, args.data_cropsize, args.data_cropsize)
                else:
                    prompt_tensor = prompt_list[i_prompt]
                # prompt_tensor, _ = ADMM(logger, prompt_list[i_prompt].cpu().numpy())  
                # 遍历prompt_list中每个张量中的每一层，张量维度为selected_clients*N*N
                for i_client in range(len(selected_clients)):
                    if i_client == 0:
                        i_prompt_new = client_weight_list[i_client] * prompt_tensor[i_client,:,:]
                    else:
                        i_prompt_new = i_prompt_new + client_weight_list[i_client] * prompt_tensor[i_client,:,:]
                # i_prompt_new /= len(selected_clients)
                global_prompt = torch.cat((global_prompt, i_prompt_new.unsqueeze(0)), dim=0)
                # global_prompt = torch.cat((global_prompt, torch.from_numpy(i_prompt_new).unsqueeze(0)),dim=0)

        global_prompt = global_prompt.unsqueeze(dim=0)


        # global model加载参数
        for k in global_para.keys():
            global_model.state_dict()[k].data.copy_(global_para[k]) 
        global_model.state_dict()['prompt_embeddings'].data.copy_(global_prompt)    

        ## 测试性能                     
        _, global_test_dataloader, _ ,_  = get_dataloader(args.dataset, 
                                                            args.data_directory, 
                                                            args.batch_size,
                                                            args.batch_size)

        globalModel_test_acc, precision, recall, fscore = compute_accuracy(global_model, global_test_dataloader, get_confusion_matrix=True,
                                                             device='cuda')
        # if (round+1)%10==0:
        #     io.savemat('E:/FL_code/FedProm/feature_matrix/FedGPT_'+args.dataset+'_1_logit_'+str(round+1)+'.mat',{'latent':inter_feature.data.cpu().numpy()})
        #     io.savemat('E:/FL_code/FedProm/feature_matrix/FedGPT_'+args.dataset+'_1_conf_'+str(round+1)+'.mat',{'conf_matrix':conf_matrix})  
                   

        logger.info("The global model's test ACC: {}, precision: {}, recall: {}, fscore: {}".format(globalModel_test_acc, precision, recall, fscore))
        logger.info("The {}-th communication completes".format(round+1))
        logger.info("************************************")
    
        # 将 global para 下发给各个客户端 
        for i_client in selected_clients: 
            for k in global_para.keys():
                local_models[i_client].state_dict()[k].data.copy_(global_para[k]) 
            local_models[i_client].state_dict()['prompt_embeddings'].data.copy_(global_prompt)
                