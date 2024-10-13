import pandas as pd
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
import myutilstrain as mut
import getopt
import sys
import pickle


def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def mse(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.square(y_true - predictions))

epochs = 5
records_per_hour = 12
forecast_hours = 24
shuffle=False
activation = 'tanh'
early_stop_patience=20
interp_type='lin'
scion_sensor=''
temporal_res = 5
early_stop_patience = 5
# Set default values
lookback_hours = 24
learning_rate = 0.001


input_directory = f'./exper-5min/'
interp_files_dir = f'./interp_sensors_5min/'

# Get command line arguments
opts, args = getopt.getopt(sys.argv[1:], "", ["lb=", "lr=", "small"])
reduced_network = False

# Parse command line arguments
for opt, arg in opts:
	if opt == "--lb":
		lookback_hours = int(arg)
	elif opt == "--lr":
		learning_rate = float(arg)
	elif opt == "--small":
		reduced_network = True

# Update variables
lookback = lookback_hours * records_per_hour
forecast = forecast_hours * records_per_hour



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


if not os.path.exists(f'{input_directory}/Ashley_HALF_{lookback}_{temporal_res}.csv'):
	mut.create_sequence_file(interp_files_dir, input_directory, 'HALF', lookback, forecast, f'Ashley_HALF_{lookback}_{temporal_res}.csv', mini, maxi, temporal_res)
else:
	print(f'file Ashley_HALF_{lookback}_{temporal_res} exists')

df = pd.read_csv(f'./exper-5min/Ashley_HALF_{lookback}_{temporal_res}.csv')
df.info()

dataLen = len(df)
X = np.array(df.iloc[:, :-1])
Y = np.array(df.iloc[:, -1].values)
print(Y[:5],'\n\n', X[0])
X = X.reshape(-1, lookback, 1)
Y = Y.reshape(-1, 1)
print('X', X.shape, type(X),'\nY',  Y.shape,  type(Y))
model_name = None

if reduced_network:
	model2 = Sequential()
	model2.add(Conv1D(filters=64, kernel_size=3, input_shape=(lookback,1)))
	model2.add(MaxPooling1D(pool_size=2))
	model2.add(Flatten())
	model2.add(Dense(64, activation='relu'))
	model2.add(Dense(1))
	optimizer = Adam(learning_rate=learning_rate)
	early_stop = tf.keras.callbacks.EarlyStopping(patience=early_stop_patience)
	model2.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
	model2.summary()

	history = model2.fit(x=X, y=Y, batch_size=1024, epochs=25, validation_split=0.3, callbacks=[early_stop])
	model_name=f'./CNN_Ashley_HALF_{lookback}_{temporal_res}min_{learning_rate}LR_REDUCED'
	model2.save(model_name)
else:
	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=3, input_shape=(lookback,1)))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Conv1D(filters=64, kernel_size=3))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dense(1))
	optimizer = Adam(learning_rate=learning_rate)
	model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
	model.summary()

	early_stop = tf.keras.callbacks.EarlyStopping(patience=early_stop_patience)
	history = model.fit(x=X, y=Y, batch_size=1024, epochs=25, validation_split=0.3, callbacks=[early_stop])
	model_name=f'./CNN_Ashley_HALF_{lookback}_{temporal_res}min_{learning_rate}LR'
	with open(f'{model_name}_history.pkl', 'wb') as file_pi:
		pickle.dump(history.history, file_pi)
	model.save(model_name)

# save memory usave
# Get memory information
memory_info = tf.config.experimental.get_memory_info('GPU:0')
with open('memory.csv', 'a') as resultcsv:
	resultcsv.write(f"{model_name},{memory_info['peak']},train\n")
print(f"Current memory usage: {memory_info['current'] / (1024**2)} MB")
print(f"Peak memory usage: {memory_info['peak'] / (1024**2)} MB")
