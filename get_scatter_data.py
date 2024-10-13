import os
import re
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


# # prepare for measuring memory
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)





models_dir = './models_bench/'
for folder in ['lstm_100un_120lb_0.0001lr_tres5', 'CNN16_Ashley_HALF_120_5min_0.001LR', 'transformer_hs2_nh2_ff8_blocks2_mlp32_dropout0.25_0.5_lookback120_forecast288_lrb0.0001_activationtanh_interplin_sensorHALF-5min.h5']:
	#print(folder)
	if 'pth' in folder:
		continue
	maes = []
	mses = []
	sensors = []
	if os.path.exists(f"scatter_plots/gtpred_{folder}.csv"):
		continue 
	# f.write('gt,pred\n')
	if 'CNN' in folder:
		lookback = int(folder.split('_')[3])
	elif 'lstm' in folder:
		print(folder)
		third = folder.split('_')[2]
		# original
		if 'ep' in third:
			lookback = int(folder.split('_')[4].split('l')[0])	
		# retrained lstms
		else:
			lookback = int(third.split('l')[0])
	elif 'SGD' in folder:
		lookback = int(folder.split('_')[3])
	# vanilla transformer
	else:
		parameters_dict = get_param_dict_transf(folder)
		lookback = int(parameters_dict['lookback'])
		# lookback = folder.split('_')[8].split('k')[-1]
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
	
	if 'transformer' not in folder:
		model = keras.models.load_model(f'./{models_dir}/{folder}')
	else:
		model.load_weights(f'{models_dir}/{folder}')
		# model.summary()
	print('\n\n', folder)
	# results = pd.DataFrame(columns=['gt', 'pred'])
	# for f in os.listdir("./test_sensors/"):
	for f in tqdm(os.listdir("./test_sensors/")):
		sensorfile = f.split('-')[0]
		# if sensorfile == scion_sensor:
		if f'{temporal_res}min' not in f:
			continue
		# print(sensorfile)
		df = pd.read_csv(f"./test_sensors/{f}", index_col=0)
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
		y_predPrime = y_pred[:, -1].reshape(-1, 1)
		print('MAE orig:', mae(targets, y_pred), 'MAE prime:', mae(targets, y_predPrime))
		# bef = len(results)
		# print('gt', targets.shape, targets.flatten().shape, 'pred', y_pred.shape,  y_pred.flatten().shape)
		# results = pd.concat([results, pd.DataFrame({'gt': targets.flatten(), 'pred': y_pred.flatten()})], ignore_index=True)
		# results = results.concat(pd.DataFrame({'gt': targets.flatten(), 'pred': y_pred.flatten()}), ignore_index=True)
		# aft = len(results)
		# print(f"before: {bef}, after: {aft}")
	# results.to_csv(f"scatter_plots/gtpred_{folder}.csv", index=False)	
	
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

	