import argparse
import os
import ast
import torch
from main_run.train import Exp_Main
import random
import numpy as np

#fix_seed = 2023
#random.seed(fix_seed)
#torch.manual_seed(fix_seed)
#np.random.seed(fix_seed)

TYPES = {0: 'Original',
         1: 'Gaussian',
         2: 'Freq-Mask',
         3: 'Freq-Mix',
         4: 'Wave-Mask',
         5: 'Wave-Mix',
         6: 'Wave-MixUp',
         7: 'StAug', 
         11: 'TimeGAN'}


parser = argparse.ArgumentParser(description='Augmentations for Time Series Forecasting')

# basic config 
parser.add_argument('--model', type=str, required=True, default='DLinear',
                    help='model name, options: [DLinear]')

# data loader 
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--percentage', type=int, default=100, help='percentage of train data')

parser.add_argument('--patience', type=int, default=12, help='early stopping patience')
# forecasting task 
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# DLinear 
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# Augmentation
parser.add_argument('--aug_type', type=int, default=0, help='0: No augmentation, 1: Gaussian Noise, 2: Frequency Masking 3: Frequency Mixing  4: Wave Masking  5: Wave Mixing  6: Wave MixUp   7: StAug ')
parser.add_argument('--aug_rate', type=float, default=0.5, help='rate for all augmentations')
parser.add_argument('--noise_level', type=float, default=0.5, help='noise level for Gaussian')
parser.add_argument('--wavelet', type=str, default='db2', help='wavelet form for DWT')
parser.add_argument('--level', type=int, default=2, help='level for DWT')
parser.add_argument('--rates', type=str, default="[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]",
                        help='List of rates as a string, e.g., "[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]"')
parser.add_argument('--nIMF', type=int, default=500, help='number of IMFs for EMD')
parser.add_argument('--mask_rate', type=float, default=0.5, help='mask rate for all augmentations')

# GAN
parser.add_argument('--hidden_dim', type=int, default=24, help='level for DWT')
parser.add_argument('--num_layer', type=int, default=3, help='level for DWT')
parser.add_argument('--iterations', type=int, default=100, help='level for DWT')
parser.add_argument('--batch_sizeg', type=int, default=64, help='level for DWT')
parser.add_argument('--module', type=str, default='gru', help='wavelet form for DWT')



args = parser.parse_args()
args.rates = ast.literal_eval(args.rates)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    mse_avg, mae_avg, rse_avg = np.zeros(args.itr), np.zeros(args.itr), np.zeros(args.itr)
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}_{}_{}'.format(
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.des, args.aug_type,args.percentage, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mse, mae, rse = exp.test(setting)
        mse_avg[ii] = mse
        mae_avg[ii] = mae
        rse_avg[ii] = rse

    f = open("result-all-" + args.des + args.data + ".txt", 'a')
    f.write('\n')
    f.write('\n')
    f.write("-------START FROM HERE-----")
    f.write("avg_" + TYPES[args.aug_type] + "  \n")
    f.write('avg mse:{}, avg mae:{} avg rse:{}  std mse:{}, std mae:{} std rse:{}'.format(mse_avg.mean(), mae_avg.mean(), rse_avg.mean(), mse_avg.std(), mae_avg.std(), rse_avg.std()))
    f.write('\n')
    f.write('\n')
    f.close()
    torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}_{}'.format(args.model,
                                                       args.data,
                                                       args.features,
                                                    args.seq_len,
                                                    args.label_len,
                                                    args.pred_len,
                                                    args.des, args.aug_type, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
