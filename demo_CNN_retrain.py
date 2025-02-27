import argparse
import torch
import torch.utils.data as Data
import torchvision
from network_947331_freeze import Network
from torch import nn
import time
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.nn import MSELoss
from torch.utils.data import DataLoader
from spectrum_dataset import SpectrumDataset
import pandas as pd
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] ="2"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(2)

BATCH_SIZE = 5

if __name__ == '__main__':
    print('Begin')
    parser = argparse.ArgumentParser(description='Train and test a network with a specific random state.')
    parser.add_argument('--random_state', type=int, required=True, help='The random state for shuffling the dataset.')
    args = parser.parse_args()

    random_state = args.random_state

    # 创建动态的 weights 文件夹名称
    weights_dir = f'weights{random_state}'

    if os.path.exists(weights_dir):
        raise Exception(f"文件夹 {weights_dir} 已经存在，请选择其他 random_state 值。")
    else:
        os.makedirs(weights_dir)
        
    train_data = SpectrumDataset("train", random_state=random_state)
    test_data = SpectrumDataset("test", random_state=random_state)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

    train_batch_num = len(train_loader)
    test_batch_num = len(test_loader)
    ############################################################
    ###这里负责找预训练好的模型
    ########这里找到mse最小的模型名称
    files = os.listdir("pre_weights_training")     ##原来是files = os.listdir("weights_pre_training")
    filename_pth = []
    for file in files:
        if file.endswith(".pth"):
            filename_pth.append(file)  ##获取文件名名称
    filename_pth.sort(key=lambda x: float(x[14:-4]))  ###去除末尾.gjf，然后按数字大小进行排序

    dir = "pre_weights_training/" + filename_pth[0]

    old_model = torch.load(dir)
    net = Network()
    net_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in old_model.state_dict().items() if k in net_dict}  ###将old——model_dict里不属于net_dict的键剔除掉
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)

    if torch.cuda.is_available():
        #net = nn.DataParallel(net)   ##支持多个卡计算
        net.cuda()
    ################
    #scaler=amp.GradScaler()
    ################
    #opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #opt = torch.optim.RMSprop(net.parameters(), lr=0.01)
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    #scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=0, last_epoch=-1)
    scheduler = ReduceLROnPlateau(opt,mode='min',factor=0.8, patience=10, verbose=False,min_lr=0,eps=1e-08)

    ##loss_func = nn.CrossEntropyLoss()
    #loss_func = nn.MSELoss()
    loss_func = nn.L1Loss()
 #####################################
    tt=[]
 ####################################
    for epoch_index in range(200):
        #if epoch_index % 20 == 0:  # 每迭代20次，更新一次学习率
        #    for params in opt.param_groups:
        #        # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.95
        #        params['lr'] *= 0.95
        #        print("lr changed to {}".format(params['lr']))
        #        # params['weight_decay'] = 0.5  # 当然也可以修改其他属性
        print(epoch_index, opt.param_groups[0]["lr"])
        st = time.time()

        torch.set_grad_enabled(True)
        net.train()
        for train_batch_index, (img_batch, label_batch,mol_num_batch,Fre_num_batch,Fre_value_batch,mol_kind_batch) in enumerate(train_loader):    ##enumerate（）函数的第一位是train_batch_index，代表着序号
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
                #label_batch = label_batch.cuda()
                mol_kind_batch =mol_kind_batch.cuda()
                #mol_num_batch=mol_num_batch.cuda()
                #Fre_num_batch=Fre_num_batch.cuda()
                #Fre_value_batch=Fre_value_batch.cuda()

            predict, w_1, w_2 = net(img_batch)
            #print(type(predict))
            #print(predict.size())
            #print(type(mol_kind_batch))
            #print(label_batch)

            #print(predict)
            #print(mol_kind_batch)

            try:
                # 打印调试信息


                # 根据 mol_kind_batch 对 predict 进行排序
                _, sorted_indices = torch.sort(mol_kind_batch.squeeze())
                sorted_preds = predict[sorted_indices]

                # 计算所有两组不同数据之间的 ReLU 插值
                loss = 0
                for i in range(len(sorted_preds)):
                    for j in range(i + 1, len(sorted_preds)):
                        #loss += torch.relu(sorted_preds[j] - sorted_preds[i])

                        loss += torch.relu(torch.relu(sorted_preds[i]) - sorted_preds[j]+9*(j-i))

                        print(i)
                        print(f"{sorted_preds[i].item()}")
                        print(j)
                        print(f" {sorted_preds[j].item()}")
                        print(f" {torch.relu(sorted_preds[i] - sorted_preds[j]+9*(j-i))}")
                        #print(f"{loss.item()}")
                #print("loss is#######################:", loss)


            except:
                #loss = torch.tensor(0.0, requires_grad=True)  # 确保 loss 变量被定义
                #print(predict)

                print("loss is???????????????:", loss)
            net.zero_grad()
            loss.backward()    ##反向传播
            opt.step()         ##反向传播求梯度


        print('(LR:%f) Time of a epoch:%.4fs' % (opt.param_groups[0]['lr'], time.time()-st))
        #scheduler.step()
        torch.set_grad_enabled(False)
        net.eval()
        total_loss = []
        total_acc = 0
        total_sample = 0

        for test_batch_index, (img_batch, label_batch,mol_num_batch,Fre_num_batch,Fre_value_batch,mol_kind_batch) in enumerate(test_loader):
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
                #label_batch = label_batch.cuda()
                mol_kind_batch = mol_kind_batch.cuda()
                #mol_num_batch=mol_num_batch.cuda()
                #Fre_num_batch=Fre_num_batch.cuda()
                #Fre_value_batch=Fre_value_batch.cuda()

            predict, w_1, w_2 = net(img_batch)
###########################################################################################
            #_, sorted_indices = torch.sort(mol_kind_batch.squeeze())
            #sorted_preds = predict[sorted_indices]

            #loss = 0
            #for i in range(len(sorted_preds)):
            #    for j in range(i + 1, len(sorted_preds)):

            #        loss += torch.relu(torch.relu(sorted_preds[i]) - sorted_preds[j]+8*(j-i))

            loss = loss_func(predict, mol_kind_batch)

            #predict = predict.argmax(dim=1)
            total_loss.append(loss)


        ###net.train()
        print("total_loss.__len__():",total_loss.__len__())
        mean_loss = sum(total_loss) / total_loss.__len__()

        scheduler.step(mean_loss.item())  ##更新学习率
        #######
        #tt.append(mean_loss.item())
        ######
        print(f"total loss: {sum(total_loss)} len: {len(total_loss)} mean loss: {mean_loss}")
        print('[Test] epoch[%d/%d] loss:%.4f' % (epoch_index, 200, mean_loss.item()))

        weight_path = f'{weights_dir}/net_mean_loss_%.4f.pth' % (mean_loss.item())
        print('Save Net weights to', weight_path, '\n')

        torch.save(net, weight_path) #包括参数和模型

        tt = [mean_loss.item()]
        loss_tt = pd.DataFrame(tt)
        loss_tt.to_csv(f'{weights_dir}/tt.csv', mode='a+', index=None, header=None)

        files = os.listdir(weights_dir)
        filename_pth = [file for file in files if file.endswith(".pth")]
        filename_pth.sort(key=lambda x: float(x[14:-4]))

        if len(filename_pth) >= 3:
            for i in range(2, len(filename_pth)):
                os.remove(f"{weights_dir}/{filename_pth[i]}")

