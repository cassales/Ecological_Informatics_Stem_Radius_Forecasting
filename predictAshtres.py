import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import getopt
import sys
import os
import time
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf
from tensorflow import keras

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def mse(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.square(y_true - predictions))


def create_sequence_file(interp_files_dir, sequence_dir, sensor, lookback, forecast, csvFile, mini, maxi, temporal_res):
	# select the plot 
	# preprocess all sensors from the selected plot and save csvs
	# read files
	# create sequence_file with correct lookback and forecast
	file_list=[]
	for f in os.listdir(interp_files_dir):
		# 01A01T010.csv
		if f"{temporal_res}min" not in f:
			continue
		sensor_name = f.split('-')[0]
		if sensor_name == sensor:
			sensorfile=f.split('.')[0]
			#file_list.append(f'{sensor}-{lookback}')
			if not os.path.isfile(f'{sequence_dir}/{sensorfile}-{temporal_res}min-{lookback}.csv'):
				print(f"\nStarting sensor {sensor} with {temporal_res}min, lb {lookback} and fc {forecast}")
				df = pd.read_csv(f"{interp_files_dir}/{f}")

				#get series to numpy, define lookback and forecast
				dataLen=len(df)
				series=np.array(df['Value_c'].values)
				series=series.reshape(dataLen, 1)

				# minmax scale series and create x and y
				# series0=series
				series=((series-mini)/(maxi-mini))
				x=series[0:dataLen-forecast]
				y=series[forecast:dataLen]

				# generate sequences
				tGen=TimeseriesGenerator(x, y, length=lookback, batch_size=dataLen+100)
				sequences=tGen[0][0][:,:,0]
				targets=tGen[0][1][:,0]
				adf = pd.DataFrame(sequences)
				adf['target']=targets
				adf.to_csv(f'{sequence_dir}/{sensorfile}-{lookback}.csv', index=False)
				print(f"created and saved sequence for sensor {sensor} with filename: {sensorfile}-{lookback}")
			else:
				print(f"File for sensor {sensorfile}-{lookback} already created!")




#Standard value for parameters
units = 5
epochs = 5
learning_rate = 0.001
shuffle=False
activation = 'tanh'
early_stop_patience=20
temporal_res=30
lookback_hours=10
forecast_hours=24
try:
	opts, args = getopt.getopt(sys.argv[1:],'hi:u:e:r:R:l:f:a:p:S:t:')
except getopt.GetoptError:
	print(f'{sys.argv[0]} -i <lin|quad> -u <units> -e <epochs> -r <lookback_hours> -l <learning_rate> -f <forecast_hours> -a <activation> -p <patience> -S <SCION sensor> -t <temporal resolution>')
	sys.exit(2)
for opt,arg in opts:
	if opt == '-h':
		print(f'{sys.argv[0]} -i <lin|quad> -u <units> -e <epochs> -r <lookback_hours> -l <learning_rate> -f <forecast_hours> -a <activation> -p <patience> -S <SCION sensor> -t <temporal resolution>')
		sys.exit(0)
	elif opt == '-i':
		interp_type = arg
	elif opt == '-u':
		units = int(arg)
	elif opt == '-e':
		epochs = int(arg)
	elif opt == '-r':
		lookback_hours = int(arg)
	elif opt == '-l':
		learning_rate = float(arg)
	elif opt == '-f':
		forecast_hours =int(arg)
	elif opt == '-a':
		activation =str(arg)
	elif opt == '-p':
		early_stop_patience=int(arg)
	elif opt == '-S':
		scion_sensor=arg
	elif opt == '-t':
		temporal_res=int(arg)

lookback=math.ceil(60.0*lookback_hours/temporal_res)
forecast=math.ceil(60.0*forecast_hours/temporal_res)


input_directory = f'/Scratch/gcassales/FF'
interp_files_dir = f'/research/repository/gcassale/FF/interp-lin-testeAsh-resolution/'
#interp_files_dir = f'/research/repository/gcassale/FF/downsample_inter_sensors_testeAsh/'
#interp_files_dir = f'/research/repository/gcassale/FF/new_downsample_inter_sensors_testeAsh/'

#csvFile = f'{input_directory}/plot{scion_plot}-{temporal_res}-{lookback}.csv'
csvFile = f'{input_directory}/{scion_sensor}-{interp_type}-{temporal_res}min-{lookback}.csv'

start_time = time.perf_counter()

#start by getting min and max among all sensor series
mini = 9999999
maxi = -9999999
for f in os.listdir(interp_files_dir):
	if f'{temporal_res}min' not in f:
		continue
	df = pd.read_csv(f"{interp_files_dir}/{f}", index_col=False)
	series=np.array(df['Value_c'].values)
	if series.min() < mini:
		mini = series.min()
	if series.max() > maxi:
		maxi = series.max()
print(f"got min and max among all sensors for temporal resolution {temporal_res}: ", mini, maxi)


# check if input file already exists or needs to be created
if not os.path.isfile(csvFile):
# if needs to be created, start by getting min and max among all sensor series
	create_sequence_file(interp_files_dir, input_directory, scion_sensor, lookback, forecast, csvFile, mini, maxi, temporal_res)
else:
	print(f"Plot file for plot {scion_sensor}-{interp_type}-{temporal_res}min-{lookback} already exists!")


finish_creating = time.perf_counter()
time_data_creation = finish_creating - start_time
print("finished creating data, time spent:",time_data_creation)

df1 = pd.read_csv(csvFile)
df = df1.iloc[:-1]
dataLen=len(df)
xdf = df1.iloc[:,:-1]
x = np.array(xdf.values)
ydf = df1.iloc[:,-1:]
y = np.array(ydf.values)
print(x.shape)
xo = x
x = x.reshape(-1,lookback,1)
yo = y


finish_reading = time.perf_counter()
time_data_read = finish_reading - finish_creating
print("finished reading data, time spent:",time_data_read)

model = keras.models.load_model(f"lstm_{units}un_{epochs}ep_{learning_rate}lr_{lookback}lb_{forecast}fc_{activation}_pat{early_stop_patience}_{interp_type}_sensorHALF_tres{temporal_res}")

y_pred = model.predict(x)
MAE=mae(y, y_pred)
MSE=mse(y, y_pred)

end_time = time.perf_counter()
print("total time:",end_time-start_time)
print("time predicting:",end_time-finish_reading)


y_std = y * (maxi - mini) + mini
y_pred_std = y_pred * (maxi - mini) + mini
MAE_R=mae(y_std, y_pred_std)
MSE_R=mse(y_std, y_pred_std)
print(f"MAE_R {MAE_R}   MSE_R {MSE_R}")

dfpred=pd.DataFrame({'gt':y_std.flatten(), 'pred':y_pred_std.flatten()})

dfpred.to_csv(f"resolution-results-smooth/lstm_{units}un_{epochs}ep_{learning_rate}lr_{lookback}lb_{forecast}fc_{activation}_pat{early_stop_patience}_{interp_type}_{scion_sensor}-{temporal_res}.csv")

with open(f'resolution-predictions-MAE-usingHALF-smoothed.csv', 'a') as resultcsv:
    resultcsv.write(f'{units},{epochs},{early_stop_patience},{learning_rate},{lookback},{forecast},{temporal_res},{activation},{interp_type},{scion_sensor},{end_time - start_time:0.6f},{time_data_creation:0.6f},{time_data_read:0.6f},{end_time - finish_reading:0.6f},{MAE},{MSE},{MAE_R},{MSE_R}\n')
