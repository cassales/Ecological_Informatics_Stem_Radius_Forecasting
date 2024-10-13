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

epochs = 100
records_per_hour = 12
forecast_hours = 24
activation = 'tanh'
temporal_res = 5
early_stop_patience = 5
input_directory = f'./exper-5min/'
interp_files_dir = f'./interp_sensors_5min/'

# Set default values
lookback_hours = 24
learning_rate = 0.001
units = 25

# Get command line arguments
opts, args = getopt.getopt(sys.argv[1:], "", ["un=", "lb=", "lr=", 'ep='])
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

# get min and max to create data if it does not exist
mini, maxi = mut.get_min_max(interp_files_dir, temporal_res)
mut.check_create(input_directory, lookback, temporal_res, interp_files_dir, mini, maxi)

# prepare X and Y arrays
X, Y = mut.prepare_data(f'{input_directory}/Ashley_HALF_{lookback}_{temporal_res}.csv', lookback, debug=True)

# declare model
model = Sequential()
model.add(LSTM(units=units, input_shape=(lookback, 1), activation=activation))
model.add(Dense(1))
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
model.summary()

# train model
model_name = f'lstm_{units}un_{lookback_hours}l_{learning_rate}lr_{temporal_res}tr'
mut.train_given_model_and_data(model, X, Y, model_name=model_name, epochs=epochs, save_model=True, save_memory=False)