import os
import time
import random
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
from sklearn.preprocessing import  MinMaxScaler



## own packages
import visu
import eva
from model import ConvLSTM
from data import load_dataset
from data.utils import split_tscv, split_sequence
from model.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')




class Forecasting():
    def __init__(self,model_name, model_id,data_path, n_past,n_future,n_feature,batch_size,epochs,learning_rate, n_predict):
        fix_seed = 2024
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

        self.root_path = ''
        self.data_path = data_path
        self.n_past = n_past
        self.n_feature = n_feature
        self.n_future = n_future
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_id = model_id
        self.n_predict = n_predict     ### number of forecasting features, default is 1, target is 'carbon_intensity_avg'
        self.model_name = model_name

        
        self.kfold = 5    ### TSCV
        self.scale = 1
        self.checkpoints = 'checkpoints/'
        self.device = 'cuda:0' if torch.cuda.is_available()  else 'cpu' 


        self.model_dict = {
            'ConvLSTM':ConvLSTM
        }
        self._build_model()




    def _get_df_(self):  
        path = os.path.join(self.root_path, self.data_path)
        df = pd.read_csv(path)
        self.df_data = df
        return self.df_data

            
    
    """
    it prepares data for time series cross-validation (TSCV) 
    """   
    ## k is the index [0,4], since we use 5 fold time series cross validation
    def ts_cv(self, k):
        df_data = self._get_df_()
        train_index_list, vali_index_list, test_index_list = split_tscv(df_data, self.kfold)
        train_set = self.df_data.iloc[train_index_list[k]]
        test_set = self.df_data.iloc[test_index_list[k]]
        vali_set = self.df_data.iloc[vali_index_list[k]]
        
        if self.scale:
            print("Normalization (MinMaxScaler) ..")
            self.scaler = MinMaxScaler()
            self.scaler.fit_transform(train_set.values)                                        ## fit on training data
            
            train_set_scaled = self.scaler.transform(train_set.values)
            vali_set_scaled = self.scaler.transform(vali_set.values)
            test_set_scaled = self.scaler.transform(test_set.values)
        
        return train_set_scaled, vali_set_scaled, test_set_scaled
    


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

        

    """
    for each fold in 5-fold TCSV, using sliding window method to split the sequence to X and y based on the 
    history horizon(n_past) and the future horizon(n_future).
    """   
    def _dataloader_tscv(self, train_set_scaled, vali_set_scaled, test_set_scaled):  
        self.x_train, self.y_train = split_sequence(train_set_scaled, self.n_past, self.n_future)
        self.x_vali, self.y_vali = split_sequence(vali_set_scaled, self.n_past, self.n_future)
        self.x_test, self.y_test = split_sequence(test_set_scaled, self.n_past, self.n_future)
        
        # reshape the y to be (batch_size, sequence_length, num_predict_features)
        self.y_train = self.y_train.reshape(self.y_train.shape[0], self.y_train.shape[1],self.n_predict)
        self.y_test = self.y_test.reshape(self.y_test.shape[0],  self.y_train.shape[1],self.n_predict)
        self.y_vali = self.y_vali.reshape(self.y_vali.shape[0], self.y_train.shape[1],self.n_predict)
        
        # converting to Tensors
        self.x_train = torch.tensor(self.x_train).float()
        self.y_train = torch.tensor(self.y_train).float()
        self.x_test = torch.tensor(self.x_test).float()
        self.y_test = torch.tensor(self.y_test).float()
        self.x_vali = torch.tensor(self.x_vali).float()
        self.y_vali = torch.tensor(self.y_vali).float()
        
        # custom class to handle X-y pairs
        self.train_dataset =load_dataset.customDataset(self.x_train, self.y_train)
        self.test_dataset = load_dataset.customDataset(self.x_test, self.y_test)
        self.vali_dataset = load_dataset.customDataset(self.x_vali, self.y_vali)

        #  Create data loaders to iterate over the datasets in batches
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = False, drop_last = True)
        self.vali_loader = DataLoader(self.vali_dataset, batch_size=self.batch_size, shuffle = False, drop_last = True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle = False, drop_last= True )
    

    def _build_model(self):  
        self.model = self.model_dict[self.model_name].MODEL(self.n_feature, self.n_past, self.n_future,self.device).float()
        # self.model = ConvLSTM.MODEL(self.n_feature, self.n_past, self.n_future,self.device).float()
        return self.model
    


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)   
        return model_optim



    def _select_criterion(self): 
        criterion = nn.MSELoss()
        return criterion
    


    def vali(self,vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():                                                       ## perform computations without tracking gradients
            for i, (x_batch, y_batch) in enumerate(vali_loader):

                x_batch = x_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                outputs = self.model(x_batch)

                outputs = outputs.detach().cpu()
                y_batch = y_batch.detach().cpu()
                
                loss = criterion(outputs,y_batch)
                total_loss.append(loss)
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
            
    
        
    def train(self,train_loss_list,vali_loss_list):
        print("train_loader: ", len(self.train_loader))
        print("vali_loader: ", len(self.vali_loader))
        print("test_loader: ",len(self.test_loader))
        
        ### this is for checkpoints combine dir of checkpoints and model_id
        path = os.path.join(self.checkpoints, self.model_id)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()                                                     

        train_steps = len(self.train_loader)
        
        early_stopping = EarlyStopping(patience = 3, verbose = True)               
        model_optimal = self._select_optimizer()   
        criterion = self._select_criterion()                            
        
        
        for epoch in range(self.epochs):     
            iter_count = 0    
            train_loss = []
            self.model.train()
            
            epoch_time = time.time()
            
            for i, (x_batch, y_batch) in enumerate(self.train_loader):
                iter_count += 1
                model_optimal.zero_grad()                                   #### (clear out)zeros out the gradients of all the parameters associated with the model
                
                x_batch = x_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                outputs = self.model(x_batch)

                loss = criterion(outputs, y_batch)                
                train_loss.append(loss.item())
                
        
                if (i + 1) % 100 == 0:                                      ### print out the loss.item() every 100 times
                    print("\titers: {0}, epoch: {1} | loss: {2:.10f}".format(i + 1, epoch + 1, loss.item()))
            
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    
                loss.backward()
                model_optimal.step()
                
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(self.vali_dataset, self.vali_loader, criterion)
            test_loss = self.vali(self.test_dataset, self.test_loader, criterion)

            train_loss_list.append(train_loss)
            vali_loss_list.append(vali_loss)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
           
            ## early stopping check the vali_loss
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optimal, epoch + 1, self.learning_rate)
        
        ## save the best model
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return  self.model
    
    def load_model(self):
        print("loading model..")
        path = os.path.join(self.checkpoints, self.model_id)
        best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))   ### if GPU is avaliable
        self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))     #### if only CPU is avaliable
        return self.model

    
    def test(self, test=1):
        if test:
            print("loading model..")
            path = os.path.join(self.checkpoints, self.model_id)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        
        preds = []
        reals = []

   
        folder_path = './test_results/'+self.model_id + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        self.model.eval()          ## set model to evaluation mode
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(self.test_loader):
                x_batch = x_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                outputs = self.model(x_batch)

                outputs = outputs.detach().cpu().numpy()
                y_batch = y_batch.detach().cpu().numpy()

                ## Inverse Transformation, values are converted back to their original form
                if self.scale:
                    shape = outputs.shape
                    outputs = self.inverse_transform(np.tile((outputs.squeeze(0)),(1,self.n_feature)))   
                    y_batch = self.inverse_transform(np.tile((y_batch.squeeze(0)),(1,self.n_feature)))
                    outputs = outputs[:,-self.n_predict].reshape(shape)     ## target is in the last column in the file  "dk_dk2_clean.csv"
                    y_batch = y_batch[:,-self.n_predict].reshape(shape)
                  
 
                pred = outputs
                real = y_batch
                
                preds.append(pred)
                reals.append(real)
                
       
                if i % 1000 == 0:
                    inputs = x_batch.detach().cpu().numpy()
                    if self.scale:
                        shape = inputs.shape
                        inputs = self.inverse_transform(inputs.squeeze(0)).reshape(shape)
                    
                    gt = np.concatenate((inputs[0, :, -1], real[0, :, -1]), axis=0) 
                    pd = np.concatenate((inputs[0, :, -1], pred[0, :, -1]), axis=0)
                    visu.visual(gt, pd, os.path.join(folder_path, str(i) + 'carbon.png'))

                    

        preds = np.array(preds)
        reals = np.array(reals)
        print("test shape:", preds.shape, reals.shape)   ### 14401(len(test_loader)), batch_size(1),n_past
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        reals = reals.reshape(-1, reals.shape[-2], reals.shape[-1])   
        print("test shape:", preds.shape, reals.shape) ### 1,24,1
        
        
        mae, mse, rmse, r2 = eva.metric(preds,reals)
        print("mse= ", mse)
        print("mae= ", mae)
        print("rmse= ", rmse)
        print("r2= ", r2)
 
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'real.npy', reals)
 
        return 
    

    def score_of_cv(self):
        summary(self.model, input_size=(self.batch_size, self.n_past, self.n_feature))
        train_loss_list=[]
        vali_loss_list = []

        folder_path = './test_results/'+self.model_id + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for k in range(self.kfold):
            print(f"This is the iteration: {k}/{self.kfold}")
            train_set_scaled, vali_set_scaled, test_set_scaled = self.ts_cv(k)
            self._dataloader_tscv(train_set_scaled, vali_set_scaled, test_set_scaled)
            self.train(train_loss_list,vali_loss_list)

        visu.plot_loss_valiloss_training(train_loss_list,vali_loss_list,folder_path)
        return 
     



from omegaconf import DictConfig, OmegaConf
import hydra
import logging

@hydra.main(version_base=None, config_path="config", config_name="carbon")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    model_name = cfg.model.name
    model_id = cfg.model.id
    data_path = cfg.data.path
    n_feature = cfg.params.n_feature
    n_past = cfg.params.n_past
    n_future = cfg.params.n_future
    batch_size = cfg.params.batch_size
    epochs = cfg.params.epochs
    learning_rate = cfg.params.learning_rate
    n_predict = cfg.params.n_predict

    ### training the model
    t = Forecasting(model_name, model_id,data_path,n_past,n_future,n_feature,batch_size,epochs,learning_rate,n_predict)
    t.score_of_cv()


    ### testing the model
    t.test()
  

if __name__ == "__main__":
    main()