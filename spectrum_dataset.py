from types import SimpleNamespace
import torch
from torch.utils.data import Dataset
import pandas as pd


class SpectrumDataset(Dataset):

    #TRAIN_NUM = 3482     ##1184/1480 6400/8000 4654/5818  3586/4483   3520/4000
    DATA_COLUMN=8000     ###输入x的长度
    def __init__(self, train_or_test: str,random_state: int =1) -> None:
        super().__init__()
        self.train_or_test = train_or_test
        self.random_state = random_state
        self.data_df = self._load_from_csv('/home/shenyx/Data/Project1_ModelTest/Experiment/transfer_azo/IR_Raman_azo.csv')
         
    def __getitem__(self, index):
        x = torch.tensor(self.data_df.iloc[index, :self.DATA_COLUMN], dtype=torch.float)
        y = torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN+5])])
        mol_num = torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN+1])])           ###分子序号
        Fre_num=torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN+2])])      ###频率序号
        Fre_value=torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN+3])])    ###频率大小
        mol_kind=torch.tensor([float(self.data_df.iloc[index, self.DATA_COLUMN])])    ###分子种类
        x=x.reshape(int(self.DATA_COLUMN/2),2)
        x=x.T
        
        return x, y, mol_num,Fre_num,Fre_value,mol_kind

    def __len__(self):
        return len(self.data_df)

    def _load_from_csv(self, fp: str):
        df = pd.read_csv(fp,header=None)  ##header=None之前没有注意
        #transfer_data_new = df.sample(frac=1, random_state=1).reset_index(drop=True)
        transfer_data_new =df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        #random_rows = transfer_data_new.sample(n=10, random_state=17)
        #total_transfer = 500
        #len_transfer=19
        # df = df.sample(1)
        if self.train_or_test == 'train':
            #train_data = transfer_data_new.iloc[0:int(total_transfer * 0.8)]
            #new_train_data=train_data.iloc[0:int(len_transfer * 0.8)]
            #new_train_data = transfer_data_new.iloc[0:10]
            new_train_data = transfer_data_new.iloc[0:10]
            return new_train_data

        elif self.train_or_test == 'test':
            #test_data = transfer_data_new.iloc[int(total_transfer * 0.8):]
            #new_test_data=test_data.iloc[0:int(len_transfer * 0.2)]
            #new_test_data = transfer_data_new.iloc[10:19]
            new_test_data = transfer_data_new.iloc[10:20]
            return new_test_data
        else:
            raise ValueError(f"传入DatasetExample的值必须是'train'或者'test'， 但却得到了'{self.train_or_test}'")


if __name__ == '__main__':
    sd = SpectrumDataset('train')
    x, y, mol_num,Fre_num,Fre_value = sd[0]
    print(x)
    print(x.shape)
    print(y)
    print(mol_num)
    print(Fre_num)
    print(Fre_value)
