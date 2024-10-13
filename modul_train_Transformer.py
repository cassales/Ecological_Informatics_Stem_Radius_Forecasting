# import pandas as pd
import numpy as np
# import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
# from tensorflow.keras.optimizers import Adam
from keras.optimizers.legacy import Adam
import myutilstrain as mut
import getopt
import sys


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

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def mse(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.square(y_true - predictions))


class CustomCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr=1e-4, max_lr=1e-3, min_lr=1e-5, warmup_steps=20, total_steps=50):
        self.initial_lr = initial_lr  # Starting learning rate (0.01)
        self.max_lr = max_lr          # Maximum learning rate (0.1)
        self.min_lr = min_lr          # Final learning rate (0.0001)
        self.warmup_steps = warmup_steps  # Number of warmup steps to reach max_lr
        self.total_steps = total_steps    # Total number of steps (decay after warmup)

    def __call__(self, step):
        # Warm-up phase: linearly increase to max_lr
        if step < self.warmup_steps:
            return self.initial_lr + (self.max_lr - self.initial_lr) * (step / self.warmup_steps)

        # Cosine decay phase after warmup
        decay_steps = self.total_steps - self.warmup_steps
        step_after_warmup = step - self.warmup_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step_after_warmup / decay_steps))
        decayed = (self.max_lr - self.min_lr) * cosine_decay + self.min_lr
        return decayed


epochs = 100
records_per_hour = 12
forecast_hours = 24
shuffle=False
temporal_res = 5
early_stop_patience = 10
input_directory = f'./exper-5min/'
interp_files_dir = f'./interp_sensors_5min/'

head_size=2
num_heads=2
num_transformer_blocks=2
mlp_units=32
mlp_dropout=0.25
dropout=0.5

# Set default values
lookback_hours = 24
learning_rate = 0.001
ff_dim=8


# Get command line arguments
opts, args = getopt.getopt(sys.argv[1:], "", ["ff=", "lb=", "lr=", "ep=", "tr="])
reduced_network = False

# Parse command line arguments
for opt, arg in opts:
	if opt == "--lb":
		lookback_hours = int(arg)
	elif opt == "--lr":
		learning_rate = float(arg)
	elif opt == "--ff":
		ff_dim = int(arg)
	elif opt == "--ep":
		epochs = int(arg)
	elif opt == "--tr":
		temporal_res = int(arg)
		records_per_hour = 60 / temporal_res
if temporal_res != 5:
	input_directory = f'./exper-downsample/'
	interp_files_dir = f'./interp-downsample/'

# Update variables
lookback = math.ceil(lookback_hours * records_per_hour)
forecast = forecast_hours * records_per_hour

# get min and max to create data if it does not exist
mini, maxi = mut.get_min_max(interp_files_dir, temporal_res)
mut.check_create(input_directory, lookback, temporal_res, interp_files_dir, mini, maxi)

# prepare X and Y arrays
X, Y = mut.prepare_data(f'{input_directory}/Ashley_HALF_{lookback}_{temporal_res}.csv', lookback, debug=True)

# declare model
model = build_model(
    (lookback,1),
    head_size=head_size,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_transformer_blocks=num_transformer_blocks,
    mlp_units=[mlp_units],
    mlp_dropout=mlp_dropout,
    dropout=dropout,
    n_pred=forecast+1
)

model.summary()

# Create the custom learning rate schedule
LR_init=learning_rate
LR_min=LR_init*10
LR_max=LR_init*10*10
# 6 4 5 or 7 5 6s
scheduler = CustomCosineDecay(initial_lr=LR_init, max_lr=LR_max, min_lr=LR_min, warmup_steps=epochs/5, total_steps=epochs)

callbacks = [
    tf.keras.callbacks.LearningRateScheduler(
        scheduler), 
    tf.keras.callbacks.EarlyStopping(
        patience=early_stop_patience,
        monitor='val_loss',
        mode='min',
        restore_best_weights=True)]


model.compile(loss='MSE', metrics=['mae', 'mse'], optimizer=Adam())


# train model
model_name = f'transformer_{ff_dim}ff_{lookback_hours}lb_{learning_rate}initlr_{temporal_res}tr'
if temporal_res == 720:
	early_stop_patience = epochs
elif temporal_res> 180:
	early_stop_patience = 50
else:
	early_stop_patience = 15
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=early_stop_patience, restore_best_weights=True)]
mut.train_given_model_and_data(model, X, Y, model_name=model_name, epochs=epochs, save_model=True, save_memory=False, callbacks=callbacks)