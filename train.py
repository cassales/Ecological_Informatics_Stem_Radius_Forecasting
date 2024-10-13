import pandas as pd
import numpy as np
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf
from tensorflow import keras

# %%
def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def mse(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.square(y_true - predictions))

# %%
units = 5
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
interp_files_dir = f'/interp_sensors_5min/'

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


# %%
import myutilstrain as mut

# %%
mut.create_sequence_file(interp_files_dir, input_directory, 'HALF', lookback, forecast, f'Ashley_HALF_{lookback}_{temporal_res}.csv', mini, maxi, temporal_res)

# %%
df = pd.read_csv('./exper-5min/Ashley_HALF_120_5.csv')
print(df.info())


# %%
dataLen = len(df)
X = np.array(df.iloc[:, :-1])
Y = np.array(df.iloc[:, -1].values)
print(Y[:5],'\n\n', X[0])
# ns=((series-mini)/(maxi-mini))
#define lookback and forecast
# x=ns[0:dataLen-forecast]
# y=ns[lookback:dataLen]
# print(len(x), x[:5], f'\n{len(y)}', y[:5])
# tGen=TimeseriesGenerator(x, y, length=lookback, batch_size=dataLen+100)
# sequences=tGen[0][0][:,:,0]
# targets=tGen[0][1][:,0]
# adf = pd.DataFrame(sequences)
# adf['target']=targets
# display(adf.head(5))
# adf.to_csv(f'testing.csv', index=False)
# # print(f"created and saved sequence for sensor {sensor} with lookback {lookback} and forecast {forecast}")

# print(adf.info())

# input_dataset = list(tf.keras.utils.timeseries_dataset_from_array(x, None, sequence_length=lookback, sequence_stride=1, batch_size=dataLen+100))
# target_dataset = list(tf.keras.utils.timeseries_dataset_from_array(y, None, sequence_length=1, sequence_stride=1, batch_size=dataLen+100))
# X = np.array(input_dataset[0]).reshape(-1, lookback, 1)
# # X = X
# Y = np.array(target_dataset[0]).reshape(-1, 1, 1)
# # Y = Y
print('X', X.shape, type(X),'\nY',  Y.shape,  type(Y))

# %%
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, input_shape=(lookback,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3))
model.add(MaxPooling1D(pool_size=2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
model.summary()

# %%
history = model.fit(x=X, y=Y, batch_size=1024, epochs=50, validation_split=0.3)

# %%
model.save(f'./CNN_Ashley_HALF_{lookback}_{temporal_res}min')
