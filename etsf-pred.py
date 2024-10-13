import pandas as pd
import numpy as np
import torch
import random
from utils.metrics import metric
import torch.nn as nn
import os
import time
import warnings
from models import ETSformer
import getopt
import math
import sys

import myutilspred as mup

# def mae(y_true, predictions):
#     y_true, predictions = np.array(y_true), np.array(predictions)
#     return np.mean(np.abs(y_true - predictions))

# def mse(y_true, predictions):
# 	y_true, predictions = np.array(y_true), np.array(predictions)
# 	return np.mean(np.square(y_true - predictions))

# TEST
# def create_sequence_file(interp_files_dir, sequence_dir, sensor, lookback, forecast, csvFile, mini, maxi, temporal_res):
# 	# select the plot (or plots)
# 	# preprocess all sensors from the selected plot and save csvs -> all files already interpolated
# 	# foreach sensor file in plot
# 	# create sequence_file with correct lookback and forecast
# 	# save csvs
# 	for f in os.listdir(interp_files_dir):
# 		if f"{temporal_res}min" not in f:
# 			continue
# 		sensor_name = f.split('-')[0]
# 		if sensor_name == sensor:
# 			sensorfile=f.split('.')[0]
# 			if not os.path.isfile(f'{sequence_dir}/{sensorfile}-{lookback}.csv'):
# 				print(f"\nstarting sensor {sensorfile} with tres {temporal_res}, lb {lookback} and fc {forecast}")
# 				df = pd.read_csv(f"{interp_files_dir}/{f}", index_col=None)
# 				#get series to numpy, define lookback and forecast
# 				datalen=len(df)
# 				print(df.columns)
# 				series=np.array(df['Value_c'].values)
# 				series=series.reshape(datalen, 1)
# 				# minmax scale series and create x and y
# 				# series0=series
# 				series=((series-mini)/(maxi-mini))
# 				x=series[0:datalen-forecast]
# 				y=series[forecast:datalen]
# 				# generate sequences
# 				tgen=TimeseriesGenerator(x, y, length=lookback, batch_size=datalen+100)
# 				sequences=tgen[0][0][:,:,0]
# 				targets=tgen[0][1][:,0]
# 				adf = pd.DataFrame(sequences)
# 				adf['target']=targets
# 				if np.any(np.isnan(adf)):
# 					print("has nan",f,len(adf))
# 					adf.dropna(inplace=True)
# 					print("after drop", len(adf))
# 				adf.to_csv(f'{sequence_dir}/{sensorfile}-{lookback}.csv', index=False)
# 				print(f"created and saved sequence for sensor {sensorfile}-{lookback} and forecast {forecast}")
# 			else:
# 				print(f"file for sensor {sensorfile}-{lookback} already created!")

class Exp_Main(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            #device = torch.device('cpu')
            #device = torch.device('mps')
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model_dict = {
            'ETSformer': ETSformer,
        }
        model = ETSformer(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _set_data(self, loaders):
        self.train_loader=loaders['train']
        self.val_loader=loaders['val']
        self.test_loader=loaders['test']
        
    
    def _get_data(self, flag):
        if flag=='train':
            data_loader=self.train_loader
        elif flag =='val':
            data_loader=self.val_loader
        elif flag =='test':
            data_loader=self.test_loader
        return data_loader

    def _select_optimizer(self):
        if 'warmup' in self.args.lradj:
            lr = self.args.min_lr
        else:
            lr = self.args.learning_rate

        if self.args.smoothing_learning_rate > 0:
            smoothing_lr = self.args.smoothing_learning_rate
        else:
            smoothing_lr = 100 * self.args.learning_rate

        if self.args.damping_learning_rate > 0:
            damping_lr = self.args.damping_learning_rate
        else:
            damping_lr = 100 * self.args.learning_rate

        nn_params = []
        smoothing_params = []
        damping_params = []
        for k, v in self.model.named_parameters():
            if k[-len('_smoothing_weight'):] == '_smoothing_weight':
                smoothing_params.append(v)
            elif k[-len('_damping_factor'):] == '_damping_factor':
                damping_params.append(v)
            else:
                nn_params.append(v)

        model_optim = Adam([
            {'params': nn_params, 'lr': lr, 'name': 'nn'},
            {'params': smoothing_params, 'lr': smoothing_lr, 'name': 'smoothing'},
            {'params': damping_params, 'lr': damping_lr, 'name': 'damping'},
        ])

        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')
        test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                #print('batch_x')
                #print(batch_x.shape)
                #print('batch_y')
                #print(batch_y.shape)
                #print('batch_x_mark')
                #print(batch_x_mark.shape)
                #print(i)
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
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
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, data, save_vals=True):
        """data - 'val' or 'test' """
        test_loader = self._get_data(flag=data)

        print('loading model',f'./{setting}/checkpoint.pth')
        self.model.load_state_dict(torch.load(os.path.join(f'./{args.checkpoints}/{setting}/checkpoint.pth')))
        #torch.load(setting,map_location=torch.device('cuda'))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                #batch_y=batch_y.squeeze()
                #print(f'batch X: {batch_x.shape} batch y: {batch_y.shape}  ')

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

	# original
        preds = np.array(preds, dtype=object)
        preds = np.concatenate([arr for arr in preds])
        trues = np.array(trues, dtype=object)
        #print(trues.shape, trues[0].shape, trues[1].shape)
        trues = np.concatenate([arr for arr in trues])
        #trues = trues.reshape(-1,1,1)
        print('test shape:', preds.shape, trues.shape)
        #print('test 0 shape:' , preds[0].shape)i
        #print('true 0 shape:' , trues[0].shape)
        #preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        #trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #print('test shape:', preds.shape, trues.shape)


        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + f'{data}_metrics.npy', np.array([mae, mse, rmse, mape, mspe]))

        if save_vals:
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
        #print('preds[0]:{}, trues[0]:{}'.format(preds[0], trues[0]))

        return preds,trues



class Object(object):
    pass

args=Object()
args.gpu='0'
args.use_gpu=True
args.use_multi_gpu=False
args.seq_len=288
args.label_len=1
args.pred_len=1
args.e_layers=2
args.d_layers=2
args.enc_in=1
args.dec_in=1
args.c_out=1
args.d_model=256
args.n_heads=2
args.d_ff=32
args.K=1
args.dropout=0.5
args.activation='sigmoid'
args.output_attention=False
args.std=0.2
args.checkpoints='cpoints'
args.patience=5
args.lradj='exponential_with_warmup'
args.min_lr=1e-30
args.smoothing_learning_rate=0
args.learning_rate=1e-5
args.damping_learning_rate=0
args.train_epochs=50
args.features='S'
args.warmup_epochs=5
args.tres = 5
args.model_id = 'test1'
args.model = 'ETSformer'
args.data = 'ForestFlows'
forecast_hours=24

try:
    opts, args2 = getopt.getopt(sys.argv[1:],'hr:l:S:t:')
except getopt.GetoptError:
    print(f'{sys.argv[0]} -r <lookback_hours> -l <learning_rate> -S <SCION plot> -t <temporal resolution>')
    sys.exit(2)
for opt,arg in opts:
    if opt == '-h':
        print(f'{sys.argv[0]} -r <lookback_hours> -l <learning_rate> -S <SCION plot> -t <temporal resolution>')
        sys.exit(0)
    elif opt == '-r':
        lookback_hours = int(arg)
    elif opt == '-l':
        args.learning_rate = float(arg)
    elif opt == '-S':
        scion_sensor=arg
    elif opt == '-t':
        args.tres=int(arg)

#lookback_hours = 10
#args.learning_rate = 0.00001
#scion_sensor='01A17T013'
#args.tres=30

lookback=math.ceil(60.0*lookback_hours/args.tres)
forecast=math.ceil(60.0*forecast_hours/args.tres)
args.seq_len=int(lookback)
print('args.seq_len', args.seq_len)

model_str=f'ETSFormer-HALFtrain-{lookback}lb-{forecast}fc-{args.learning_rate}lr-{args.tres}min.pth'
if (model_str not in os.listdir()):
	print(f'{model_str} does not exist!')
	exit(0)

#input_directory = f'/Users/gwcassales/Documents/ETSF/'
input_directory = f'/Scratch/gcassales/FF'
#interp_files_dir = f'/research/repository/gcassale/FF/downsample_interp_sensors/'
interp_files_dir = f'/research/repository/gcassale/FF/interp-lin-testeAsh-resolution/'
#interp_files_dir = f'/Users/gwcassales/Documents/SCION-ML/interp-lin-testeAsh-resolution/'
# interp_files_dir = f'/research/repository/gcassale/FF/interp-lin-trainAsh-resolution/'

start_time = time.perf_counter()

#start by getting min and max among all sensor series
# mini = 9999999
# maxi = -9999999
# for f in os.listdir(interp_files_dir):
# 	if f'{args.tres}min' not in f:
# 		continue
# 	df = pd.read_csv(f"{interp_files_dir}/{f}", index_col=False)
# 	series=np.array(df['Value_c'].values)
# 	if series.min() < mini:
# 		mini = series.min()
# 	if series.max() > maxi:
# 		maxi = series.max()
mini, maxi = mup.get_min_max(interp_files_dir, args.tres)
print(f"got min and max among all sensors for temporal resolution {args.tres}: ", mini, maxi)
# check if input file already exists or needs to be created

csvFile = f'{input_directory}/{scion_sensor}-lin-{args.tres}min-{lookback}.csv'
if not os.path.isfile(csvFile):
	# creates the file
	mup.create_sequence_file(interp_files_dir, input_directory, scion_sensor, lookback, forecast, csvFile, mini, maxi, args.tres)
else:
	print(f"Sensor file for plot {scion_sensor} with {lookback} lookback and {args.tres}min resolution already exists!")

finish_creating = time.perf_counter()
time_data_creation = finish_creating - start_time
print("finished creating data, time spent:",time_data_creation)

exp=Exp_Main(args)

# train data
# df=pd.read_csv('/Scratch/gcassales/FF/plotHALF-lin-288.csv')
df=pd.read_csv(csvFile)



from torch.utils.data import DataLoader,Dataset

class numpyDataset(Dataset):
    def __init__(self, x,y,x_mark,y_mark):
        super(Dataset, self).__init__()
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)
        self.x_mark = torch.tensor(x_mark) 
        self.y_mark = torch.tensor(y_mark)
        
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, index):
        return self.x[index].unsqueeze(-1), self.y[index].unsqueeze(-1),self.x_mark[index].unsqueeze(-1),self.y_mark[index].unsqueeze(-1)

seq_len=args.seq_len
pred_len=1
#print('seq_len',seq_len, 'pred_len', pred_len, 'columns', df.columns)
x=df.iloc[:,0:seq_len]
y=df.iloc[:,seq_len:seq_len+pred_len]
#print(y)
data_len=len(x)
x_mark=np.zeros((data_len,seq_len))
y_mark=np.zeros((data_len,pred_len))
# train_len=int(0.7*data_len)
print('data len', data_len)


train_dataset=numpyDataset(x.values,y.values,x_mark,y_mark)
val_dataset=numpyDataset(x.values,y.values,x_mark,y_mark)
# test data
# need to load different file and use testrun
test_dataset=numpyDataset(x.values,y.values,x_mark,y_mark)
train_loader=DataLoader(train_dataset,batch_size=1024)
val_loader=DataLoader(val_dataset,batch_size=1024)
test_loader=DataLoader(test_dataset,batch_size=1024)
loader={}
loader['train']=train_loader
loader['val']=val_loader
loader['test']=test_loader

#items=next(iter(loader['test']))
#print('batch_X',items[0])
#
#print('batch_y',items[1])
#print('batch_Xm',items[2])
#print('batch_ym',items[3])
exp._set_data(loader)

finish_reading = time.perf_counter()
time_data_read = finish_reading - finish_creating
print("finished reading data, time spent:",time_data_read)

# train_dataset.__len__()
test_dataset.__len__()

# exp.train('folder1')

# torch.save(exp.model, 'ETSFormer-HALFtrain-288lb-05dpout.pth')
# print('loading model')
# exp.model = torch.load(model_str,map_location=torch.device('cpu'))
y_pred, gt = exp.test(f'train-{model_str[:-4]}',data='test', save_vals=True)
# exp.test(setting, data='test', save_vals=True)
#print('before reshape',y_pred.shape, gt.shape)
#y_pred = y_pred[:0].reshape(len(y_pred),-1)
#print('after reshape', y_pred.shape, gt.shape)

MAE=mup.mae(gt,y_pred)
MSE=mup.mse(gt,y_pred)
y_std = gt * (maxi - mini) + mini
y_pred_std = y_pred * (maxi - mini) + mini

MAE_R=mup.mae(y_std,y_pred_std)
MSE_R=mup.mse(y_std,y_pred_std)
print(f'MAE {MAE} MSE {MSE} MAE_R {MAE_R} MSE_R {MSE_R}')
dfpred=pd.DataFrame({'gt': y_std.flatten(), 'pred': y_pred_std.flatten()})
dfpred.to_csv(f"HALF-train-results-smooth-rescale/{model_str[:-4]}-{scion_sensor}.csv")

dfpred=pd.DataFrame({'gt': gt.flatten(), 'pred': y_pred.flatten()})
dfpred.to_csv(f"HALF-train-results-smooth-nores/{model_str[:-4]}-{scion_sensor}.csv")


end_time = time.perf_counter()
total_time = end_time-start_time
predicting_time = end_time-finish_reading
print("total time:", total_time)
print("time predicting:", predicting_time)
# print('epochs run',len(history.history['loss']))

with open(f'ETSFormer-pred-resolution-smoothed.csv', 'a') as resultcsv:
	resultcsv.write(f'{args.learning_rate},{args.seq_len},{forecast},{args.tres},{scion_sensor},{total_time:0.6f},{time_data_creation:0.6f},{time_data_read:0.6f},{predicting_time:0.6f},{MAE},{MSE},{MAE_R},{MSE_R}\n')
