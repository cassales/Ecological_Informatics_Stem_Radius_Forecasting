# %%
import pandas as pd
import numpy as np
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
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
epochs = 5
records_per_hour = 12
learning_rate = 0.001
forecast_hours = 24
lookback_hours = 36
shuffle=False
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


# %%
import myutilstrain as mut

# %%
if not os.path.exists(f'{input_directory}/Ashley_HALF_{lookback}_{temporal_res}.csv'):
	# os.remove(f'Ashley_HALF_{lookback}_{temporal_res}.csv')
	mut.create_sequence_file(interp_files_dir, input_directory, 'HALF', lookback, forecast, f'Ashley_HALF_{lookback}_{temporal_res}.csv', mini, maxi, temporal_res)
else:
	print(f'file Ashley_HALF_{lookback}_{temporal_res}.csv already exists')

# %%
df = pd.read_csv(f'./exper-5min/Ashley_HALF_{lookback}_{temporal_res}.csv')
df.info()


# %%
dataLen = len(df)
X = np.array(df.iloc[:, :-1])
Y = np.array(df.iloc[:, -1].values)
print(Y[:5],'\n\n', X[0])
X = X.reshape(-1, lookback, 1)
Y = Y.reshape(-1, 1)
print('X', X.shape, type(X),'\nY',  Y.shape,  type(Y))
#
## %%
#model = Sequential()
#model.add(Conv1D(filters=64, kernel_size=3, input_shape=(lookback,1)))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Conv1D(filters=64, kernel_size=3))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(1))
#optimizer = Adam(learning_rate=0.001)
#model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
#model.summary()
#
## %%
#history = model.fit(x=X, y=Y, batch_size=64, epochs=50, validation_split=0.3)
#
## %%
#model.save(f'./CNN_Ashley_HALF_{lookback}_{temporal_res}min_0.001LR')

# %%
model2 = Sequential()
model2.add(Conv1D(filters=64, kernel_size=3, input_shape=(lookback,1)))
model2.add(MaxPooling1D(pool_size=2))
model2.add(Conv1D(filters=64, kernel_size=3, input_shape=(lookback,1)))
model2.add(MaxPooling1D(pool_size=2))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(1))
optimizer = Adam(learning_rate=0.01)
model2.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
model2.summary()

history = model2.fit(x=X, y=Y, batch_size=64, epochs=50, validation_split=0.3)

model2.save(f'./CNN_Ashley_HALF_{lookback}_{temporal_res}min_0.01LR')

# %%
#model3 = Sequential()
#model3.add(Conv1D(filters=64, kernel_size=3, input_shape=(lookback,1)))
#model3.add(MaxPooling1D(pool_size=2))
#model3.add(Dense(64, activation='relu'))
#model3.add(Dense(1))
#optimizer = Adam(learning_rate=0.01)
#model3.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
#model3.summary()
#
#history = model3.fit(x=X, y=Y, batch_size=1024, epochs=50, validation_split=0.3)
#
#model3.save(f'./CNN_Ashley_HALF_{lookback}_{temporal_res}min_0.01LR_REDUCED')
#
## %%
#model4 = Sequential()
#model4.add(Conv1D(filters=64, kernel_size=3, input_shape=(lookback,1)))
#model4.add(MaxPooling1D(pool_size=2))
#model4.add(Dense(64, activation='relu'))
#model4.add(Dense(1))
#optimizer = Adam(learning_rate=0.001)
#model4.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
#model4.summary()
#
#history = model4.fit(x=X, y=Y, batch_size=1024, epochs=50, validation_split=0.3)
#
#model4.save(f'./CNN_Ashley_HALF_{lookback}_{temporal_res}min_0.001LR_REDUCED')
#
# %%




