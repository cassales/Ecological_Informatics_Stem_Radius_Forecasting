import os
import re
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
# from sklearn.metrics import mean_absolute_error, mean_squared_error
import myutilstrain as mut
from tqdm import tqdm


def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def mse(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.square(y_true - predictions))

def build_model(algorithm):
	if algorithm == 'LinReg':
		return LinearRegression()
	elif algorithm == 'SVN':
		return LinearSVR(dual=False)
	else:
		raise ValueError('Invalid algorithm')

interp_files_dir = './interp_sensors_5min/'
input_directory = './exper-5min/'
temporal_res = 5
forecast = 288

mini = 9999999
maxi = -9999999
for f in os.listdir(interp_files_dir):
	sensorfile = f.split('-')[0]
	df = pd.read_csv(f"{interp_files_dir}/{f}", index_col=0)
	# print(len(df))
	series=np.array(df['Value_c'].values)
	if series.min() < mini:
		mini = series.min()
	if series.max() > maxi:
		maxi = series.max()
print("got min and max among all sensors", mini, maxi)


# models_dir = './models_bench/'
# for folder in os.listdir(f'./{models_dir}/'):
for lookback in [120, 288, 576]:
	# load data
	if not os.path.exists(f'{input_directory}/Ashley_HALF_{lookback}_{temporal_res}.csv'):
		mut.create_sequence_file(interp_files_dir, input_directory, 'HALF', lookback, forecast, f'Ashley_HALF_{lookback}_{temporal_res}.csv', mini, maxi, temporal_res)
	else:
		print(f'\n\nfile Ashley_HALF_{lookback}_{temporal_res}.csv exists')

	df = pd.read_csv(f'./{input_directory}/Ashley_HALF_{lookback}_5.csv')
	df.info()

	dataLen = len(df)
	print(dataLen)
	X = np.array(df.iloc[:, :-1])
	Y = np.array(df.iloc[:, -1].values)
	# print(Y[:5],'\n\n', X[0])

	X = X.reshape(-1, lookback, 1)
	Y = Y.reshape(-1, 1)
	Xprime = X.reshape(X.shape[0], -1)

	for alg in ['SVN', 'LinReg']:
		model = build_model(alg)
		model.fit(Xprime, Y.ravel())
		
		maes = []
		mses = []
		sensors = []
		if os.path.exists(f"metrics_per_model/metrics_{alg}_{lookback}.csv"):
			continue 
		
		# for f in os.listdir("./test_sensors/"):
		print(alg, lookback)
		for f in tqdm(os.listdir("./test_sensors/")):
			sensorfile = f.split('-')[0]
			# if sensorfile == scion_sensor:
			if f'{temporal_res}min' not in f:
				continue
			# print(sensorfile)
			df = pd.read_csv(f"./test_sensors/{f}", index_col=0)
			series=np.array(df['Value_c'].values)
			dataLen = len(series)
			ns=((series-mini)/(maxi-mini))
			x=ns[0:dataLen-forecast]
			y=ns[forecast:dataLen]
			# print(len(x), len(y))
			tGen=TimeseriesGenerator(x, y, length=lookback, batch_size=dataLen+100)
			# break
			sequences=tGen[0][0].reshape(-1, lookback)
			targets=tGen[0][1].reshape(-1,1)
			# print(sequences.shape, targets.shape)
			y_pred = model.predict(sequences)
			sensors.append(sensorfile)
			maes.append(mae(targets, y_pred))
			mses.append(mse(targets, y_pred))
			# print('MAE:', mae_val, 'MSE:', mse_val, '\n\n')
			# break
		print('\n\n------------------------------------',f'\n{alg}_{lookback}')
		print('MAE:', np.mean(maes), 'MSE:', np.mean(mses), '\n\n')
		with open(f"metrics_per_model/metrics_{alg}_{lookback}.csv", 'w') as f:
			f.write('sensor,mae,mse\n')
			for sensor, mae_val, mse_val in zip(sensors,maes,mses):
				f.write(f"{sensor},{mae_val},{mse_val}\n")
		
	# # save memory usave
	# # Get memory information
	# memory_info = tf.config.experimental.get_memory_info('GPU:0')
	# with open('memory-test.csv', 'a') as resultcsv:
	# 	resultcsv.write(f"{folder},{memory_info['peak']},test\n")
	
	# print(f"Current memory usage: {memory_info['current'] / (1024**2)} MB")
	# print(f"Peak memory usage: {memory_info['peak'] / (1024**2)} MB")
	# # Clear the Keras session to free up memory
	# keras.backend.clear_session()
	# print(f"Current memory usage: {memory_info['current'] / (1024**2)} MB")
	# # Reset memory info for the next model
	# tf.config.experimental.reset_memory_stats('GPU:0')

	