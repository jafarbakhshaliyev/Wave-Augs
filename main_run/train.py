# =============================================================================
# The code is originated from
# Chen, M., Xu, Z., Zeng, A., & Xu, Q. (2023). "FrAug: Frequency Domain Augmentation for Time Series Forecasting".
# arXiv preprint arXiv:2302.09292.
# =============================================================================

import os
import torch
import torch.nn.functional as F
import numpy as np

from dataset_loader.datasetloader import data_provider
from models import DLinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import torch
import torch.nn as nn
from torch import optim
import time
import warnings
import matplotlib.pyplot as plt
from augmentation.aug import augmentation


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

warnings.filterwarnings('ignore')

TYPES = {0: 'None',
         1: 'Freq-Mask',
         2: 'Freq-Mix',
         3: 'Wave-Mask',
         4: 'Wave-Mix',
         5: 'StAug'}

class Exp_Main(Exp_Basic):

    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
         
        model_dict = {
            'DLinear': DLinear
         }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):

        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        time_now = time.time()

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        time_now = time.time()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

    
            for i, (batch_x, batch_y, aug_data) in enumerate(train_loader):


                iter_count += 1
                model_optim.zero_grad()
                if self.args.aug_type == 5:
                    aug_data = aug_data.float().to(self.device)
                else:
                    aug_data = None
                
                if self.args.aug_type:
                  aug = augmentation()
                  if self.args.aug_type == 1:
                    xy = aug.freq_mask(batch_x, batch_y[:, -self.args.pred_len:, :], rate=self.args.aug_rate, dim=1)
                    batch_x2, batch_y2 = xy[:, :self.args.seq_len, :], xy[:, -self.args.label_len-self.args.pred_len:, :]
                    batch_x = torch.cat([batch_x,batch_x2],dim=0)
                    batch_y = torch.cat([batch_y,batch_y2],dim=0)
                  elif self.args.aug_type  == 2:
                    xy = aug.freq_mix(batch_x, batch_y[:, -self.args.pred_len:, :], rate=self.args.aug_rate, dim=1)
                    batch_x2, batch_y2 = xy[:, :self.args.seq_len, :], xy[:, -self.args.label_len-self.args.pred_len:, :]
                    batch_x = torch.cat([batch_x,batch_x2],dim=0)
                    batch_y = torch.cat([batch_y,batch_y2],dim=0)
                  elif self.args.aug_type == 3:
                    xy = aug.wave_mask(batch_x, batch_y[:, -self.args.pred_len:, :] ,rates = self.args.rates, wavelet =self.args.wavelet, level = self.args.level, dim = 1)
                    batch_x2, batch_y2 = xy[:, :self.args.seq_len, :], xy[:, -self.args.label_len-self.args.pred_len:, :]
                    sampling_steps = int(batch_x2.shape[0] * self.args.sampling_rate)
                    indices = torch.randperm(batch_x2.shape[0])[:sampling_steps]
                    batch_x2 = batch_x2[indices,:,:]
                    batch_y2 = batch_y2[indices,:,:]
                    batch_x = torch.cat([batch_x,batch_x2],dim=0)
                    batch_y = torch.cat([batch_y,batch_y2],dim=0)
                  elif self.args.aug_type == 4:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    xy = aug.wave_mix(batch_x, batch_y[:, -self.args.pred_len:, :] ,rates = self.args.rates, wavelet = self.args.wavelet, level = self.args.level, dim = 1)
                    batch_x2, batch_y2 = xy[:, :self.args.seq_len, :], xy[:, -self.args.label_len-self.args.pred_len:, :]
                    sampling_steps = int(batch_x2.shape[0] * self.args.sampling_rate)
                    indices = torch.randperm(batch_x2.shape[0])[:sampling_steps]
                    batch_x2 = batch_x2[indices,:,:]
                    batch_y2 = batch_y2[indices,:,:]
                    batch_x = torch.cat([batch_x,batch_x2],dim=0)
                    batch_y = torch.cat([batch_y,batch_y2],dim=0)
                  elif self.args.aug_type == 5:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    weighted_xy = aug.emd_aug(aug_data)
                    weighted_x, weighted_y = weighted_xy[:,:self.args.seq_len,:], weighted_xy[:,-self.args.label_len-self.args.pred_len:,:]
                    batch_x, batch_y = aug.mix_aug(weighted_x, weighted_y, lambd = self.args.aug_rate)
                     

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:] 
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) 
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        min_val_loss = early_stopping.get_val_loss_min()
        
        return self.model, min_val_loss

    def test(self, setting,  test=1):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  #
                true = batch_y  

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                    
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()

        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f = open(self.args.des + self.args.data + ".txt", 'a')
        f.write(" \n")
        f.write('{} --- Pred {} -> mse:{}, mae:{}, rse:{}'.format(TYPES[self.args.aug_type], self.args.pred_len, mse, mae, rse))
        f.write('\n')
        f.close()

        return mse, mae, rse    
