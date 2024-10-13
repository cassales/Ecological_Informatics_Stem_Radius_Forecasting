import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
import myutilstrain as mut
import getopt
import sys


def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def mse(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.square(y_true - predictions))

# def train_given_model_and_data(model, X, Y, model_name=None, epochs=100, save_history=False):
# 	# prepare for measuring memory
# 	gpus = tf.config.experimental.list_physical_devices('GPU')
# 	if gpus:
# 		try:
# 			for gpu in gpus:
# 				tf.config.experimental.set_memory_growth(gpu, True)
# 		except RuntimeError as e:
# 			print(e)
	
# 	early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
# 	history = model.fit(x=X, y=Y, batch_size=1024, epochs=epochs, validation_split=0.3, callbacks=[early_stop])

# 	if model_name is None:
# 		model_name = "testing"
	
# 	if save_history:
# 		with open(f'{model_name}_history.pkl', 'wb') as file_pi:
# 			pickle.dump(history.history, file_pi)
# 	model.save(model_name)
	
# 	# save memory usave
# 	# Get memory information
# 	memory_info = tf.config.experimental.get_memory_info('GPU:0')
# 	with open('memory.csv', 'a') as resultcsv:
# 		resultcsv.write(f"{model_name},{memory_info['peak']},train\n")
# 	print(f"Current memory usage: {memory_info['current'] / (1024**2)} MB")
# 	print(f"Peak memory usage: {memory_info['peak'] / (1024**2)} MB")

epochs = 100
records_per_hour = 12
forecast_hours = 24
shuffle=False
activation = 'tanh'
temporal_res = 5
early_stop_patience = 5

# Set default values
lookback_hours = 24
learning_rate = 0.001
units = 25

input_directory = f'./exper-5min/'
interp_files_dir = f'./interp_sensors_5min/'

# Get command line arguments
opts, args = getopt.getopt(sys.argv[1:], "", ["un=", "lb=", "lr=", 'ep='])
# opts, args = getopt.getopt(sys.argv[1:], "", ["lb=", "lr=", "small"])
reduced_network = False

# Parse command line arguments
for opt, arg in opts:
	if opt == "--lb":
		lookback_hours = int(arg)
	elif opt == "--lr":
		learning_rate = float(arg)
	elif opt == "--un":
		units = int(arg)
	elif opt == "--ep":
		epochs = int(arg)

# Update variables
lookback = lookback_hours * records_per_hour
forecast = forecast_hours * records_per_hour



# start_time = time.perf_counter()
# mini = 9999999
# maxi = -9999999
# for f in os.listdir(interp_files_dir):
# 	sensorfile = f.split('-')[0]
# 	df = pd.read_csv(f"{interp_files_dir}/{f}", index_col=0)
# 	series=np.array(df['Value_c'].values)
# 	if series.min() < mini:
# 		mini = series.min()
# 	if series.max() > maxi:
# 		maxi = series.max()
# print("got min and max among all sensors", mini, maxi)
mini, maxi = mut.get_min_max(interp_files_dir, temporal_res)

mut.check_create(input_directory, lookback, temporal_res, interp_files_dir, mini, maxi)

# if not os.path.exists(f'{input_directory}/Ashley_HALF_{lookback}_{temporal_res}.csv'):
# 	mut.create_sequence_file(interp_files_dir, input_directory, 'HALF', lookback, forecast, f'Ashley_HALF_{lookback}_{temporal_res}.csv', mini, maxi, temporal_res)
# else:
# 	print(f'file Ashley_HALF_{lookback}_{temporal_res} exists')

# df = pd.read_csv(f'./exper-5min/Ashley_HALF_{lookback}_{temporal_res}.csv')
# df.info()

# dataLen = len(df)
# X = np.array(df.iloc[:, :-1])
# Y = np.array(df.iloc[:, -1].values)
# X = X.reshape(-1, lookback, 1)
# Y = Y.reshape(-1, 1)
# print('X', X.shape, type(X),'\nY',  Y.shape,  type(Y))
X, Y = mut.prepare_data(f'{input_directory}/Ashley_HALF_{lookback}_{temporal_res}.csv', lookback, debug=True)

model = Sequential()
model.add(LSTM(units=units, input_shape=(lookback, 1), activation=activation))
model.add(Dense(1))
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
model.summary()

model_name = f'lstm_{units}un_{lookback_hours}l_{learning_rate}lr_{temporal_res}tr'
mut.train_given_model_and_data(model, X, Y, model_name=model_name, epochs=epochs, save_model=False, save_memory=False)
