import argparse 

def get_args():
    parser = argparse.ArgumentParser()
    ## flowers102
    parser.add_argument('--model', type=str, default='imagenet_sup_rn18', help='基础模型')
    parser.add_argument('--dataset', type=str, default='cifar10', help='数据集')
    parser.add_argument('--classes', type=int, default=10, help='数据集种类数目')
    parser.add_argument('--partition_style', type=str, default='noniid', help='划分数据集的方式')
    parser.add_argument('--batch_size', type=int, default=8, help='batch大小')  
    parser.add_argument('--beta', type=float, default=1, help='迪利克雷分布的超参')
    parser.add_argument('--ratio', type=float, default=0.5, help='保存奇异值的比例')
    parser.add_argument('--first_round', type=int, default=40, help='前多少轮进行t-SVD分解')
    parser.add_argument('--sigma', type=float, default=100, help='高斯核函数中的带宽')
    
    parser.add_argument('--adam_lr', type=float, default=0.01, help='adam学习率')

    parser.add_argument('--weight_decay', type=float, default=1e-5, help='学习率衰减率') 

    parser.add_argument('--n_clients', type=int, default=20, help='客户端个数')
    parser.add_argument('--n_selected_clients', type=int, default=10, help='被选择客户端个数')

    parser.add_argument('--epochs', type=int, default=5, help='模型训练次数')
    parser.add_argument('--commucation_times', type=int, default=50, help='通信次数')

    parser.add_argument('--data_directory', type=str, default='E:/FL_code/FL-KD/', help='数据存储路径')
    parser.add_argument('--log_directory', type=str, default='E:/FL_code/FedProm/LOG', help='日志文件路径')
    parser.add_argument('--device', type=str, default='cuda:2', help='指定GPU') 
    # parser.add_argument('--out_dim', type=int, default=256, help='resnet输出的特征向量维度') 
    
    ##prompt相关参数
    parser.add_argument('--model_transfer_type', type=str, default='prompt', help='模型训练类型,e.g.,end2end,prompt,adapter')
    parser.add_argument('--model_prompt_location', type=str, default='below', help='prompt插入的位置,e.g.,below or pad')
    parser.add_argument('--model_prompt_num_tokens', type=int, default=6, help='插入prompt的个数')
    parser.add_argument('--model_prompt_initiation', type=str, default='random', help='初始化的方法')
    parser.add_argument('--data_cropsize', type=int, default=224, help='裁剪尺寸')
    parser.add_argument('--mlp_dimension', type=list, default=[512])

    args = parser.parse_args()
    return args 

