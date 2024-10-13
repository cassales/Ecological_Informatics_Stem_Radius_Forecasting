import os
import re
import math
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import layers
import tensorflow as tf
from tqdm import tqdm

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def mse(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.square(y_true - predictions))

def get_param_dict_transf(folder):
	parameters = re.findall(r'_([^_]+)', folder)
	# Initialize an empty dictionary
	parameters_dict = {}
	# Loop through each parameter and separate the key and value
	for param in parameters:
		# Use regex to separate the word (key) from the number (value)
		match = re.match(r'([a-zA-Z]+)([\d\.\-]+)', param)
		if match:
			key = match.group(1)  # The word part
			value = match.group(2)  # The numeric part
			parameters_dict[key] = value
		else:
			# Handle cases where no number is present (like 'interplin')
			parameters_dict[param] = None

	return parameters_dict

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="tanh")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0,n_pred=1):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="tanh")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_pred)(x)
    return keras.Model(inputs, outputs)

interp_files_dir = './interp_sensors_5min/'
temporal_res = 5
forecast = 288

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

models_dir = './models_bench/'
for folder in os.listdir(f'./{models_dir}/'):
	if 'CNN16' in folder and '120' in folder and '5min' in folder:
		print(folder)
	if 'pth' in folder:
		continue
	if 'CNN16_Ashley_HALF_120_5min_0.001LR' not in folder:
		continue
	maes = []
	mses = []
	sensors = []
	if os.path.exists(f"metrics_per_model/metrics_{folder}2.csv"):
		continue 
	if 'CNN' in folder:
		lookback = int(folder.split('_')[3])
		temporal_res = int(folder.split('_')[4].split('m')[0])
	elif 'lstm' in folder:
		print(folder)
		third = folder.split('_')[2]
		# original
		if 'ep' in third:
			lookback = int(folder.split('_')[4].split('l')[0])
			temporal_res = int(folder.split('_')[-1].split('s')[-1])
		# retrained lstms
		else:
			lookback_hours = int(third.split('l')[0])
			temporal_res = int(folder.split('_')[-1].split('t')[0])
			lookback = lookback = math.ceil(lookback_hours * (60 / temporal_res))
	elif 'SGD' in folder:
		lookback = int(folder.split('_')[3])
		temporal_res = int(folder.split('_')[-1].split('m')[0])
	# vanilla transformer
	elif 'transformer' in folder:
		if '.h5' in folder:
			parameters_dict = get_param_dict_transf(folder)
			lookback = int(parameters_dict['lookback'])
			temporal_res = int(folder.split('-')[-1].split('m')[0])
			model = build_model(
				(lookback, 1),
				head_size=int(parameters_dict['hs']),
				num_heads=int(parameters_dict['nh']),
				ff_dim=int(parameters_dict['ff']),
				num_transformer_blocks=int(parameters_dict['blocks']),
				mlp_units=[int(parameters_dict['mlp'])],
				mlp_dropout=0.25,
				dropout=0.5,
				n_pred=int(parameters_dict['forecast'])+1
			)
		# new save format not using h5
		else:
			lookback_hours = int(folder.split('_')[2].split('l')[0])
			temporal_res = int(folder.split('_')[-1].split('t')[0])
			lookback = math.ceil(lookback_hours * (60 / temporal_res))
	
	# load model/weights
	if '.h5' not in folder:
		model = keras.models.load_model(f'./{models_dir}/{folder}')
	else:
		model.load_weights(f'{models_dir}/{folder}')

	# load each test file, forecast, and calculate metrics
	print('\n\n------------------------------------',f'\n{folder}')

	for f in tqdm(os.listdir("./test_sensors/")):
		sensorfile = f.split('-')[0]
		# if sensorfile == scion_sensor:
		# if f'{temporal_res}min' not in f:
		# 	continue
		# print(sensorfile)
		df = pd.read_csv(f"./test_sensors/{f}", index_col=0)
		# set DATE to datetime resample on the spot based on temporal_res value
		df.index = pd.to_datetime(df.index)
		df = df.resample(f'{temporal_res}Min').first()
		series=np.array(df['Value_c'].values)
		dataLen = len(series)
		colu = len(df.columns)
		# print(dataLen,colu,forecast,lookback)
		ns=((series-mini)/(maxi-mini))
		x=ns[0:dataLen-forecast]
		y=ns[forecast:dataLen]
		# print(len(x), len(y))
		tGen=TimeseriesGenerator(x, y, length=lookback, batch_size=dataLen+100)
		# break
		sequences=tGen[0][0].reshape(-1, lookback, 1)
		targets=tGen[0][1].reshape(-1,1)
		# print(sequences.shape, targets.shape)
		y_pred = model.predict(sequences, verbose=0)
		sensors.append(sensorfile)
		maes.append(mae(targets, y_pred))
		mses.append(mse(targets, y_pred))
		# print('MAE:', mae_val, 'MSE:', mse_val, '\n\n')
		# break
	print('MAE:', np.mean(maes), 'MSE:', np.mean(mses), '\n\n')
	with open(f"metrics_per_model/metrics_{folder}2.csv", 'w') as f:
		f.write('sensor,mae,mse\n')
		for sensor, mae_val, mse_val in zip(sensors,maes,mses):
			f.write(f"{sensor},{mae_val},{mse_val}\n")
