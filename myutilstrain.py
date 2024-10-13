import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pandas as pd
import os
import pickle

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def mse(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.square(y_true - predictions))

def check_create(input_directory, lookback, temporal_res, interp_files_dir, mini, maxi, forecast=288):
	if not os.path.exists(f'{input_directory}/Ashley_HALF_{lookback}_{temporal_res}.csv'):
		create_sequence_file(interp_files_dir, input_directory, 'HALF', lookback, forecast, f'Ashley_HALF_{lookback}_{temporal_res}.csv', mini, maxi, temporal_res)
	else:
		print(f'file Ashley_HALF_{lookback}_{temporal_res} exists')

def should_process(fname, plot, temporal_res):
	ret = True
	if f'{temporal_res}min' not in fname:
		print(f'skip {fname}, different resolution')
		return False
	sensor_plot = fname.split('A')[1].split('T')[0]
	if plot != 'FULL':
		if plot == 'HALF':
			ret = (int(sensor_plot) <= 13)
		else:
			ret = (int(sensor_plot) == plot) 
	return ret

def create_sequence_file(interp_files_dir, sequence_dir, plot, lookback, forecast, csvFile, mini, maxi, temporal_res):
	# select the plot (or plots)
	# preprocess all sensors from the selected plot and save csvs -> all files already interpolated
	# foreach sensor file in plot
	# create sequence_file with correct lookback and forecast
	# save csvs
	file_list=[]
	for f in os.listdir(interp_files_dir):
		#print("got file",f,"starting processing")
		# should_process has the logic according to the types of possible scion_plot 
		if should_process(f, plot, temporal_res):
			sensor=f.split('.')[0]
			file_list.append(f'{sensor}-{lookback}')
			if not os.path.isfile(f'{sequence_dir}/{sensor}-{lookback}.csv'):
				#print(f"\n{sequence_dir}/{sensor}-{lookback}.csv does not exist. Creating file for sensor {sensor}")
				#print(f"reading {f}")
				df = pd.read_csv(f"{interp_files_dir}/{f}", index_col=False)

				#get series to numpy, define lookback and forecast
				datalen=len(df)
				if datalen < forecast+lookback:
					print(f"datalen too small. skipping {f}")
					continue
				series=np.array(df['Value_c'].values)
				series=series.reshape(datalen, 1)

				# minmax scale series and create x and y
				# series0=series
				series=((series-mini)/(maxi-mini))
				x=series[0:datalen-forecast]
				y=series[forecast:datalen]

				# generate sequences
				tgen=TimeseriesGenerator(x, y, length=lookback, batch_size=datalen+100)
				sequences=tgen[0][0][:,:,0]
				targets=tgen[0][1][:,0]
				adf = pd.DataFrame(sequences)
				adf['target']=targets
				if np.any(np.isnan(adf)):
					print("has nan",f,len(adf))
					adf.dropna(inplace=True)
					print("after drop", len(adf))
				adf.to_csv(f'{sequence_dir}/{sensor}-{lookback}.csv', index=False)
				print(f"created and saved sequence for sensor {sensor} with lookback {lookback} and forecast {forecast}")
			else:
				print(f"file for sensor {sensor}-{lookback} already created!")

	# concatenate all sensorfiles
	import subprocess
	print(file_list)
	with open(f'{sequence_dir}/{csvFile}', 'w') as out_file:
		bashcommand = f"cat {sequence_dir}/{file_list[0]}.csv"
		subprocess.run(bashcommand.split(' '), stdout=out_file)
		for file in file_list[1:]:
			bashcommand = f"tail -n +2 {sequence_dir}/{file}.csv"
			subprocess.run(bashcommand.split(' '), stdout=out_file)


def create_sequence_file_seq_forecast(interp_files_dir, sequence_dir, plot, lookback, forecast, csvFile, mini, maxi, temporal_res):
	# select the plot (or plots)
	# preprocess all sensors from the selected plot and save csvs -> all files already interpolated
	# foreach sensor file in plot
	# create sequence_file with correct lookback and forecast
	# save csvs
	file_list=[]
	for f in os.listdir(interp_files_dir):
		#print("got file",f,"starting processing")
		# should_process has the logic according to the types of possible scion_plot 
		if should_process(f, plot, temporal_res):
			sensor=f.split('.')[0]
			file_list.append(f'{sensor}-{lookback}')
			if not os.path.isfile(f'{sequence_dir}/{sensor}-{lookback}.csv'):
				#print(f"\n{sequence_dir}/{sensor}-{lookback}.csv does not exist. Creating file for sensor {sensor}")
				#print(f"reading {f}")
				df = pd.read_csv(f"{interp_files_dir}/{f}", index_col=False)

				#get series to numpy, define lookback and forecast
				datalen=len(df)
				if datalen < forecast+lookback:
					print(f"datalen too small. skipping {f}")
					continue
				series=np.array(df['Value_c'].values)
				series=series.reshape(datalen, 1)

				# minmax scale series and create x and y
				# series0=series
				series=((series-mini)/(maxi-mini))
				x=series[0:datalen-forecast]
				y=series[lookback:datalen]

				# generate sequences
				input_dataset = list(tf.keras.utils.timeseries_dataset_from_array(x, None, sequence_length=lookback, sequence_stride=1, batch_size=datalen+100))
				target_dataset = list(tf.keras.utils.timeseries_dataset_from_array(y, None, sequence_length=forecast, sequence_stride=1, batch_size=datalen+100))
				X = np.array(input_dataset[0]).reshape(-1, lookback)
				Y = np.array(target_dataset[0]).reshape(-1, forecast)
				adf = pd.concat([pd.DataFrame(X),pd.DataFrame(Y)], axis=1)
				adf.columns = range(adf.columns.size)
				# sanitization
				if np.any(np.isnan(adf)):
					print("has nan",f,len(adf))
					adf.dropna(inplace=True)
					print("after drop", len(adf))
				adf.to_csv(f'{sequence_dir}/{sensor}-{lookback}.csv', index=False)
				print(f"created and saved sequence for sensor {sensor} with lookback {lookback} and forecast {forecast}")
			else:
				print(f"file for sensor {sensor}-{lookback} already created!")

	# concatenate all sensorfiles
	import subprocess
	print(file_list)
	with open(csvFile, 'w') as out_file:
		bashcommand = f"cat {sequence_dir}/{file_list[0]}.csv"
		subprocess.run(bashcommand.split(' '), stdout=out_file)
		for file in file_list[1:]:
			bashcommand = f"tail -n +2 {sequence_dir}/{file}.csv"
			subprocess.run(bashcommand.split(' '), stdout=out_file)
			bashcommand = f"rm {sequence_dir}/{file}.csv"
			subprocess.run(bashcommand.split(' '), stdout=out_file)


def get_min_max(interp_files_dir, temporal_res):
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
	return mini, maxi

def train_given_model_and_data(model, X, Y, batch_size=1024, model_name=None, epochs=100, save_history=False, save_model=True, save_memory=True, shuffle=False, callbacks=None):
	if save_memory:
		# prepare for measuring memory
		gpus = tf.config.experimental.list_physical_devices('GPU')
		if gpus:
			try:
				for gpu in gpus:
					tf.config.experimental.set_memory_growth(gpu, True)
			except RuntimeError as e:
				print(e)
	if callbacks is None:
		early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
		callbacks = [early_stop]
	history = model.fit(x=X, y=Y, batch_size=batch_size, epochs=epochs, shuffle=shuffle, validation_split=0.3, callbacks=callbacks)

	if model_name is None:
		model_name = "testing"
	
	if save_history:
		with open(f'{model_name}_history.pkl', 'wb') as file_pi:
			pickle.dump(history.history, file_pi)
	
	if save_model and epochs > 1:
		model.save(model_name)
	
	if save_memory:
		# save memory usave
		# Get memory information
		memory_info = tf.config.experimental.get_memory_info('GPU:0')
		with open('memory.csv', 'a') as resultcsv:
			resultcsv.write(f"{model_name},{memory_info['peak']},train\n")
		print(f"Current memory usage: {memory_info['current'] / (1024**2)} MB")
		print(f"Peak memory usage: {memory_info['peak'] / (1024**2)} MB")

def prepare_data(csv_file, lookback, debug=False):
	df = pd.read_csv(csv_file)
	df.info()
	X = np.array(df.iloc[:, :-1])
	Y = np.array(df.iloc[:, -1].values)
	X = X.reshape(-1, lookback, 1)
	Y = Y.reshape(-1, 1)
	if debug:
		print('X', X.shape, type(X),'\nY',  Y.shape,  type(Y))
	return X, Y