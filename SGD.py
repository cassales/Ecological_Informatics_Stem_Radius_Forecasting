import myutilstrain as mut
import pandas as pd
import numpy as np
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, InputLayer
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def mse(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.square(y_true - predictions))


epochs = 5
records_per_hour = 12
learning_rate = 0.001
forecast_hours = 24
lookback_hours = 10
shuffle=False
lookback_hours = 10
activation = 'tanh'
early_stop_patience=20
interp_type='lin'
scion_sensor=''
temporal_res = 5

input_directory = f'./exper-5min/'
interp_files_dir = f'./interp_sensors_5min/'

lookback=lookback_hours * records_per_hour
# lookback=10
forecast=forecast_hours * records_per_hour



# start_time = time.perf_counter()
mini = 9999999
maxi = -9999999
for f in os.listdir(interp_files_dir):
	sensorfile = f.split('-')[0]
	# print(f)
	# if sensorfile == scion_sensor:
	# if f'{temporal_res}min' not in f:
	# 	continue
	df = pd.read_csv(f"{interp_files_dir}/{f}", index_col=0)
	# print(len(df))
	series=np.array(df['Value_c'].values)
	if series.min() < mini:
		mini = series.min()
	if series.max() > maxi:
		maxi = series.max()
print("got min and max among all sensors", mini, maxi)


# prepare for measuring memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
	except RuntimeError as e:
		print(e)



# for lookback in [120, 288, 288*2]:
for lookback in [288*2]:
	# create model
	model = Sequential()
	model.add(InputLayer((lookback,1)))
	model.add(Flatten())
	model.add(Dense(1))
	optimizer = Adam(learning_rate=0.001)
	model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
	model.summary()


	# load data
	if not os.path.exists(f'{input_directory}/Ashley_HALF_{lookback}_{temporal_res}.csv'):
		mut.create_sequence_file(interp_files_dir, input_directory, 'HALF', lookback, forecast, f'Ashley_HALF_{lookback}_{temporal_res}.csv', mini, maxi, temporal_res)
	else:
		print(f'file Ashley_HALF_{lookback}_{temporal_res}.csv exists')

	df = pd.read_csv(f'./exper-5min/Ashley_HALF_{lookback}_5.csv')
	df.info()

	dataLen = len(df)
	X = np.array(df.iloc[:, :-1])
	Y = np.array(df.iloc[:, -1].values)
	print(Y[:5],'\n\n', X[0])

	X = X.reshape(-1, lookback, 1)
	Y = Y.reshape(-1, 1)
	print('X', X.shape, type(X),'\nY',  Y.shape,  type(Y)) 

	# train model
	history = model.fit(X, Y, epochs=10, verbose=1, shuffle=True, validation_split=0.3)
	model_name=f'SGD_Ashley_HALF_{lookback}_5min'
	model.save(model_name)

	print(f"saved model {model_name}")

	# save memory usave
	# Get memory information
	memory_info = tf.config.experimental.get_memory_info('GPU:0')
	with open('memory.csv', 'a') as resultcsv:
		resultcsv.write(f"{model_name},{memory_info['peak']},train\n")
	
	print(f"Current memory usage: {memory_info['current'] / (1024**2)} MB")
	print(f"Peak memory usage: {memory_info['peak'] / (1024**2)} MB")
	# Clear the Keras session to free up memory
	print("\nCleaning session and resetting memory stats\n------------------------------------")
	keras.backend.clear_session()
	tf.config.experimental.reset_memory_stats('GPU:0')
	print(f"Current memory usage: {memory_info['current'] / (1024**2)} MB")
	print(f"Peak memory usage: {memory_info['peak'] / (1024**2)} MB")
