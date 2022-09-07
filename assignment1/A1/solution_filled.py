# This script contains the helper functions you will be using for this assignment

import os
import random

import numpy as np
import torch
import h5py
#from torch._C import float32
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class BassetDataset(Dataset):
    """
    BassetDataset class taken with permission from Dr. Ahmad Pesaranghader

    We have already processed the data in HDF5 format: er.h5
    See https://www.h5py.org/ for details of the Python package used

    We used the same data processing pipeline as the paper.
    You can find the code here: https://github.com/davek44/Basset
    """

    # Initializes the BassetDataset
    def __init__(self, path='./data/', f5name='er.h5', split='train', transform=None):
        """
        Args:
            :param path: path to HDF5 file
            :param f5name: HDF5 file name
            :param split: split that we are interested to work with
            :param transform (callable, optional): Optional transform to be applied on a sample
        """

        self.split = split

        split_dict = {'train': ['train_in', 'train_out'],
                      'test': ['test_in', 'test_out'],
                      'valid': ['valid_in', 'valid_out']}

        assert self.split in split_dict, "'split' argument can be only defined as 'train', 'valid' or 'test'"

        # Open hdf5 file where one-hoted data are stored
        self.dataset = h5py.File(os.path.join(path, f5name.format(self.split)), 'r')

        # Keeping track of the names of the target labels
        self.target_labels = self.dataset['target_labels']

        # Get the list of volumes
        self.inputs = self.dataset[split_dict[split][0]]
        self.outputs = self.dataset[split_dict[split][1]]

        self.ids = list(range(len(self.inputs)))
        if self.split == 'test':
            self.id_vars = np.char.decode(self.dataset['test_headers'])

    def __getitem__(self, i):
        """
        Returns the sequence and the target at index i

        Notes:
        * The data is stored as float16, however, your model will expect float32.
          Do the type conversion here!
        * Pay attention to the output shape of the data.
          Change it to match what the model is expecting
          hint: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        * The target must also be converted to float32
        """

        idx = self.ids[i]

        # Sequence & Target
        output = {'sequence': None, 'target': None}

        # WRITE CODE HERE
        output["sequence"]=torch.tensor(self.inputs[idx],dtype=torch.float32)
        output["sequence"]=output["sequence"].permute((1,2,0))
        output["target"]=torch.tensor(self.outputs[idx],dtype=torch.float32)
        return output

    def __len__(self):
        # WRITE CODE HERE
        return len(self.inputs)

    def get_seq_len(self):
        """
        Answer to Q1 part 2
        """
        # WRITE CODE HERE
        return self.inputs.shape[3]


    def is_equivalent(self):
        """
        Answer to Q1 part 3
        """
        # WRITE CODE HERE
        if (self.inputs.shape[1] ==4 and self.inputs.shape[2] 
        ==1 and self.inputs.shape[3]==self.get_seq_len()):
          return True
        return False
        
class Basset(nn.Module):
    """
    Basset model
    Architecture specifications can be found in the supplementary material
    You will also need to use some Convolution Arithmetic
    """

    def __init__(self):
        super(Basset, self).__init__()

        self.dropout = 0.3
        self.num_cell_types = 164

        self.conv1 = nn.Conv2d(1, 300, (19, 4), stride=(1, 1), padding=(9, 0))
        self.conv2 = nn.Conv2d(300, 200, (11, 1), stride=(1, 1), padding=(5, 0))
        self.conv3 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(4, 0))

        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)
        self.maxpool1 = nn.MaxPool2d((3, 1))#defual kernel size
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))

        self.fc1 = nn.Linear(13*200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)

        self.fc3 = nn.Linear(1000, self.num_cell_types)
        self.m=nn.Dropout(0.3)

    def forward(self, x):

        # WRITE CODE HERE

        x=self.conv1(x)
        x=self.bn1(x) 
        x=F.relu(x)
        x=self.maxpool1(x)

        x=self.conv2(x)
        x=self.bn2(x) 
        x=F.relu(x)
        x=self.maxpool2(x)

        x=self.conv3(x)
        x=self.bn3(x) 
        x=F.relu(x)
        x=self.maxpool3(x)
        # print('after 3rd cnn layer:\t{}'.format(x.size()))

        x= x.view(x.size()[0], -1)
        x = self.fc1(x)
        x=self.bn4(x)
        x=F.relu(x)
    
        # print('no drop',torch.sum(torch.nonzero(x)))
        x=self.m(x)
        
        # print('after 1st FC layer:\t{}'.format(x.size()))
        # print('after 1st drop',torch.sum(torch.nonzero(x)))

        x = self.fc2(x)
        x=self.bn5(x)
        x=F.relu(x)
        # print('no drop',torch.sum(torch.nonzero(x)))
        x=self.m(x)
        # print('after 2rd drop',torch.sum(torch.nonzero(x)))
        # print('after 2rd FC layer:\t{}'.format(x.size()))

        x = self.fc3(x)
        # x = F.sigmoid(x)
        # print('after 3rd FC layer:\t{}'.format(x.size()))
        # print(x)
        return x


def compute_fpr_tpr(y_true, y_pred):
    """
    Computes the False Positive Rate and True Positive Rate
    Args:
        :param y_true: groundtruth labels (np.array of ints)
        :param y_pred: model decisions (np.array of ints)

    :Return: dict with tpr, fpr (values are floats)
    """
    output = {'fpr': 0., 'tpr': 0.}
    # WRITE CODE HERE
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))

    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    output["fpr"]=fp/(fp + tn)
    output["tpr"]=tp/(tp + fn)
    # print(f'tp/(tp + fn)',tp/(tp + fn))
    # print(f'fp/(fp + tn)',fp/(fp + tn))
    # print(f'fn',fn)
    return output


def compute_fpr_tpr_dumb_model():
    """
    Simulates a dumb model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05 
                 ...
            Do the same for output['tpr_list']
             
    """
    # WRITE CODE HERE

    k = np.arange(0, 1, 0.05)
    output = {'fpr_list': [], 'tpr_list': []}
    # import matplotlib.pyplot as plt


    y_true=[]
    y_pred=[]
    for a in range(1000):
        y_true.append(random.randint(0, 1))
        y_pred.append(random.uniform(0, 1))
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)

    for i in range(len(k)):
        y_pred_binary=np.where(y_pred >= k[i], 1, 0)
        # print(k[i])
        # print(compute_fpr_tpr(y_true,y_pred_binary))
        # fpr,tpr=compute_fpr_tpr(y_true,y_pred_binary).values()
        fpr=compute_fpr_tpr(y_true,y_pred_binary).get("fpr")
        tpr=compute_fpr_tpr(y_true,y_pred_binary).get("tpr")
        output["fpr_list"].append(fpr)
        output["tpr_list"].append(tpr)

    return output


def compute_fpr_tpr_smart_model():
    """
    Simulates a smart model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05 
                 ...
            Do the same for output['tpr_list']
    """

    # WRITE CODE HERE
    k = np.arange(0, 1, 0.05)
    output = {'fpr_list': [], 'tpr_list': []}
    # import matplotlib.pyplot as plt


    y_true=[]
    y_pred=[]

    for a in range(1000):
        y_true.append(random.randint(0, 1))
    for i in range(len(y_true)):
        if (y_true[i]==1):
            pred=random.uniform(0.4, 1)
        else:
            pred=random.uniform(0, 0.6)
        y_pred.append(pred)

    y_true=np.array(y_true)
    y_pred=np.array(y_pred)

    for i in range(len(k)):
        y_pred_binary=np.where(y_pred >= k[i], 1, 0)
        # print(k[i])
        # print(compute_fpr_tpr(y_true,y_pred_binary).values())
        # fpr,tpr=compute_fpr_tpr(y_true,y_pred_binary).values()

        fpr=compute_fpr_tpr(y_true,y_pred_binary).get("fpr")
        tpr=compute_fpr_tpr(y_true,y_pred_binary).get("tpr")
        output["tpr_list"].append(tpr)
        output["fpr_list"].append(fpr)

    # plt.plot(output["fpr_list"])
    # plt.plot(output["tpr_list"])

    return output


def compute_auc_both_models():
    """
    Simulates a dumb model and a smart model and computes the AUC of both

    :Return: dict with auc_dumb_model, auc_smart_model.
             These contain the AUC for both models
             auc values in the lists should be floats
    """
    output = {'auc_dumb_model': 0., 'auc_smart_model': 0.}
    # WRITE CODE HERE
    k = np.arange(0, 1, 0.05)

    y_true_dumb=[]
    y_pred_dumb=[]
    for a in range(1000):
        y_true_dumb.append(random.randint(0, 1))
        y_pred_dumb.append(random.uniform(0, 1))
    y_true_dumb=np.array(y_true_dumb)
    y_pred_dumb=np.array(y_pred_dumb)

    y_true_smart=[]
    y_pred_smart=[]

    for a in range(1000):
        y_true_smart.append(random.randint(0, 1))
    for i in range(len(y_true_smart)):
        if (y_true_smart[i]==1):
            pred=random.uniform(0.4, 1)
        else:
            pred=random.uniform(0, 0.6)
        y_pred_smart.append(pred)

    y_true_smart=np.array(y_true_smart)
    y_pred_smart=np.array(y_pred_smart)
    
    output["auc_dumb_model"]=compute_auc(y_true_dumb,y_pred_dumb).get("auc")
    output["auc_smart_model"]=compute_auc(y_true_smart,y_pred_smart).get("auc")

    return output


def compute_auc_untrained_model(model, dataloader, device):
    """
    Computes the AUC of your input model
    Args:
        :param model: solution.Basset()
        :param dataloader: torch.utils.data.DataLoader
                           Where the dataset is solution.BassetDataset
        :param device: torch.device
    :Return: dict with key 'auc'.
             This contains the AUC for the model
             auc value should be float
    Notes:
    * Dont forget to re-apply your output activation!
    * Make sure this function works with arbitrarily small dataset sizes!
    * You should collect all the targets and model outputs and then compute AUC at the end
      (compute time should not be as much of a consideration here)
    """
    # WRITE CODE HERE

    output = {'auc': 0.}
    model.eval()
    with torch.no_grad():
        Y_PRED=[]
        Y_TRUE=[]
        # for i in range(100):
        # for i in range(100):
        
            # try:
            #     batch = next(iter(dataloader))
            # except StopIteration:
            #     iterloader = iter(dataloader)
            #     batch = next(iterloader)

            # print("iteration" + str(i))
            # print(batch["sequence"].shape)

        for i, batch in enumerate(dataloader):
        # for batch[i] in dataloader: 
            X=batch["sequence"].to(device)
            y=batch["target"]
            logis=model(X).to(device)
            y_pred=torch.sigmoid(logis).flatten().cpu().numpy()
            y=y.flatten().cpu().numpy()
            print(y.shape)

            Y_PRED.append(y_pred)
            Y_TRUE.append(y)
            print(f'Y_PRED',i, np.shape(Y_PRED))
            print(f'Y_TRUE',i, np.shape(Y_TRUE))
            # if i>4:
            #     break
            # print(f'Y_PRED',y_pred.shape)
        Y_PRED=np.float32(np.concatenate(Y_PRED, axis=None).flatten())
        print(Y_PRED[0].dtype)    

        Y_TRUE=np.concatenate(Y_TRUE, axis=None).flatten().astype(int)
        print(Y_TRUE[0].dtype)    

        output["auc"]=compute_auc(Y_TRUE,Y_PRED).get("auc")
        print(f'output["auc"]',output["auc"])
    # model.eval()
    # with torch.no_grad():
    #     Y_PRED=[]
    #     Y_TRUE=[]
    #     # for i in range(100):
    #     for batch in dataloader: 
    #         X=batch["sequence"].to(device)
    #         # print(X.shape)
    #         y=batch["target"]
    #         logis=model(X).to(device)
    #         # print(logis.shape)
    #         y_pred=torch.sigmoid(logis).flatten().cpu().numpy()
    #         #torch.max(logis, dim=-1)[1]
    #         # print(y_pred.dtype)
    #         y=y.flatten().cpu().numpy()
    #         # print(y.dtype)
    #     Y_PRED.append(y_pred)
    #     Y_TRUE.append(y)
    #     # print(f'Y_PRED',y_pred.shape)
    #     Y_PRED=np.array(Y_PRED)    
    #     Y_TRUE=np.array(Y_TRUE)  
    #     output["auc"]=compute_auc(Y_TRUE,Y_PRED).get("auc")
    return output

def compute_auc(y_true, y_model):
    """
    Computes area under the ROC curve (using method described in main.ipynb)
    Args:
        :param y_true: groundtruth labels (np.array of ints [0 or 1])
        :param y_model: model outputs (np.array of float32 in [0, 1])
    :Return: dict with key 'auc'.
             This contains the AUC for the model
             auc value should be float
    Note: if you set y_model as the output of solution.Basset, 
    you need to transform it before passing it here!
    """
    output = {'auc': 0.}
    # WRITE CODE HERE
    unique_prob_thresholds=np.unique(y_model)
    print(f'unique_prob_thresholds.shape',unique_prob_thresholds.shape)
    y_model=np.float32(y_model)
    print(f'y_model.shape',y_model.shape)

    # def roc_from_scratch(y_true, y_model):
    #     roc = np.array([])

    #     for i in range(len(unique_prob_thresholds)):           
    #         sorted_pred_binary=np.where(y_model<unique_prob_thresholds[i], 0, 1)
    #         fpr=compute_fpr_tpr(y_true,sorted_pred_binary).get("fpr")
    #         tpr=compute_fpr_tpr(y_true,sorted_pred_binary).get("tpr")      
    #         roc = np.append(roc, [fpr, tpr])   
    #     return roc.reshape(-1, 2)

    # ROC = roc_from_scratch(y_true,y_model)
    # fpr, tpr = ROC[:, 0], ROC[:, 1]
    TPR=[]
    FPR=[]
    k = np.arange(0, 1, 0.05)
    # for i in range(len(unique_prob_thresholds)):     
    for i in range(len(k)):           
        sorted_pred_binary=np.where(y_model<k[i], 0, 1)
        fpr=compute_fpr_tpr(y_true,sorted_pred_binary).get("fpr")
        tpr=compute_fpr_tpr(y_true,sorted_pred_binary).get("tpr")      
        TPR.append(tpr)
        FPR.append(fpr)
    # return TPR FPR
    rectangle_roc_left = 0
    rectangle_roc_right=0
    rectangle_roc=0
    FPR=np.sort(FPR)
    TPR=np.sort(TPR)
    print(f"fpr",FPR)
    print(f"tpr",TPR)

    # rectangle_roc=np.trapz(tpr,fpr)
    # d_fpr=np.diff(fpr)
    # d_fpr=np.pad(d_fpr,(0,1),"constant")
    # # d_tpr=np.pad(np.diff(tpr),(0,1),"constant")
    # d_tpr=np.diff(tpr)
    # d_tpr=np.pad(d_tpr,(0,1),"constant")
    # rectangle_roc=np.dot(tpr,d_fpr)+np.dot(d_fpr,d_tpr)/2


    for k in range(len(FPR)-1):#no use of threshold. but fpr.
        # if k>=1:

        rectangle_roc_left=(FPR[k+1]- FPR[k])*TPR[k]

        # # print(f"unique_prob_thresholds-k",unique_prob_thresholds[k])
        # # print(f"fpr-k-1",k-1,fpr[k-1])
        # # print(f"tpr-k-1",k-1,tpr[k-1])
        # # print(f"fpr-k",k,fpr[k])
        # # print(f"tpr-k",k,tpr[k])
        # print(f'rectangle_roc_left',rectangle_roc_left)
        rectangle_roc_right=(FPR[k+1]- FPR[k])*TPR[k+1]
        # print(f'rectangle_roc_right',rectangle_roc_right)
        rectangle_roc =rectangle_roc+1/2*(rectangle_roc_left+rectangle_roc_right)
        print(f'rectangle_roc',rectangle_roc)

        # rectangle_roc = rectangle_roc + 0.5*(FPR[k]- FPR[k-1]) * (TPR[k]+TPR[k-1])
    output["auc"] =rectangle_roc
    # output["auc"]=(-1)*np.trapz(TPR,FPR) 
    print(output["auc"])

    # print("sklearn-auc-score",roc_auc_score(y_true,y_model))
    return output


def get_critereon():
    """
    Picks the appropriate loss function for our task
    criterion should be subclass of torch.nn
    """

    # WRITE CODE HERE
    critereon = torch.nn.BCEWithLogitsLoss()
    return critereon


def train_loop(model, train_dataloader, device, optimizer, criterion):
    """
    One Iteration across the training set
    Args:
        :param model: solution.Basset()
        :param train_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to record losses or scores within the, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!
    """

    output = {'total_score': 0.,
              'total_loss': 0.}
     # WRITE CODE HERE
    model.train()
    Y_PRED=[]
    Y_TRUE=[]
    total_loss=0
        # for i in range(100):
    for batch in train_dataloader: 
        X=batch["sequence"].to(device)
        # print(X.shape)
        y=batch["target"]
        logis=model(X).to(device)
        logis_auc=logis.flatten().detach().cpu().numpy()
        # print(logis.shape)
        # y_pred=torch.sigmoid(logis).flatten().detach().cpu().numpy()
        #torch.max(logis, dim=-1)[1]
        # print(y_pred.dtype)
        y=y.to(device)
        y_auc=y.flatten().detach().cpu().numpy()
        loss = criterion(logis, y)
        # print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # print(y.dtype)
    Y_PRED.append(logis_auc)
    Y_TRUE.append(y_auc)
    # print(f'Y_PRED',y_pred.shape)
    Y_PRED=np.array(Y_PRED)    
    Y_TRUE=np.array(Y_TRUE)  
    output['total_score']=compute_auc(Y_TRUE,Y_PRED).get("auc")
    output['total_loss']=total_loss
    return output['total_score'], output['total_loss']


def valid_loop(model, valid_dataloader, device, optimizer, criterion):
    """
    One Iteration across the validation set
    Args:
        :param model: solution.Basset()
        :param valid_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to record losses or scores within the, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE
    model.eval()
    Y_PRED=[]
    Y_TRUE=[]
    total_loss=0
        # for i in range(100):
    for batch in valid_dataloader: 
        X=batch["sequence"].to(device)
        # print(X.shape)
        y=batch["target"]
        logis=model(X).to(device)
        logis_auc=logis.flatten().detach().cpu().numpy()

        y=y.to(device)
        y_auc=y.flatten().detach().cpu().numpy()
        loss = criterion(logis, y)
        # print(loss)

        total_loss += loss.item()

        # print(y.dtype)
    Y_PRED.append(logis_auc)
    Y_TRUE.append(y_auc)
    # print(f'Y_PRED',y_pred.shape)
    Y_PRED=np.array(Y_PRED)    
    Y_TRUE=np.array(Y_TRUE)  
    output['total_score']=compute_auc(Y_TRUE,Y_PRED).get("auc")
    output['total_loss']=total_loss
    return output['total_score'], output['total_loss']
