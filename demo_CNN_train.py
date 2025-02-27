import torch
import torch.utils.data as Data
import torchvision
from network import Network
from torch import nn
import time
import random
import numpy as np
from spectrum_dataset import SpectrumDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] ="0"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(2)

BATCH_SIZE = 16

if __name__ == '__main__':

    print('Begin')


    train_data = SpectrumDataset("train")
    test_data = SpectrumDataset("test")

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

    train_batch_num = len(train_loader)
    test_batch_num = len(test_loader)

    net = Network()
    if torch.cuda.is_available():
        #net = nn.DataParallel(net)   ##支持多个卡计算
        net.cuda()

    
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(opt,mode='min',factor=0.8, patience=10, verbose=False,min_lr=0,eps=1e-08)

    ##loss_func = nn.CrossEntropyLoss()
    loss_func = nn.MSELoss()
 #####################################
    tt=[]
 ####################################
    for epoch_index in range(200):

        print(epoch_index, opt.param_groups[0]["lr"])
        st = time.time()

        torch.set_grad_enabled(True)
        net.train()
        for train_batch_index, (img_batch, label_batch,mol_num_batch,Fre_num_batch,Fre_value_batch,mol_kind_batch) in enumerate(train_loader):    ##enumerate（）函数的第一位是train_batch_index，代表着序号
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()


            predict, w_1, w_2 = net(img_batch)

            try:
                loss = loss_func(predict, label_batch)
            except:
                print(predict)

            net.zero_grad()
            loss.backward()    ##反向传播
            opt.step()         ##反向传播求梯度


        print('(LR:%f) Time of a epoch:%.4fs' % (opt.param_groups[0]['lr'], time.time()-st))

        torch.set_grad_enabled(False)
        net.eval()
        total_loss = []
        total_acc = 0
        total_sample = 0

        for test_batch_index, (img_batch, label_batch,mol_num_batch,Fre_num_batch,Fre_value_batch,mol_kind_batch) in enumerate(test_loader):
            if torch.cuda.is_available():
                img_batch = img_batch.cuda()
                label_batch = label_batch.cuda()


            predict, w_1, w_2 = net(img_batch)
            loss = loss_func(predict, label_batch)

            predict = predict.argmax(dim=1)


            total_loss.append(loss)

            total_sample += img_batch.size(0)

        ###net.train()

        mean_loss = sum(total_loss) / total_loss.__len__()

        scheduler.step(mean_loss.item())  ##更新学习率

        print(f"total loss: {sum(total_loss)} len: {total_loss.__len__()} mean loss: {mean_loss}")
        print('[Test] epoch[%d/%d] loss:%.4f'
            % (epoch_index, 200, mean_loss.item()))

        weight_path = '/results/netr_mean_loss_%.4f.pth'%(mean_loss.item())
        print('Save Net results to', weight_path,'\n')

        net.cuda()
        torch.save(net, weight_path)
        
        tt=[mean_loss.item()]
        loss_tt=pd.DataFrame(tt)
        loss_tt.to_csv(r'tt.csv',mode='a+',index=None,header=None)

###################################################################

        files = os.listdir("/results")
        filename_pth = []
        for file in files:
            if file.endswith(".pth"):
                filename_pth.append(file)  ##获取文件名名称
        filename_pth.sort(key=lambda x: float(x[15:-4]))  
        if len(filename_pth)>=6:
            for i in range(5,len(filename_pth)):
                os.remove("/results/"+filename_pth[i])

