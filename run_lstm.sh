#!/bin/bash

# Your bash script code here
python lstm_train.py --lb 10 --lr 0.01 --ep 50 --un 25
python lstm_train.py --lb 10 --lr 0.001 --ep 50 --un 25
python lstm_train.py --lb 10 --lr 0.01 --ep 50 --un 50
python lstm_train.py --lb 10 --lr 0.001 --ep 50 --un 50
python lstm_train.py --lb 10 --lr 0.01 --ep 50 --un 100
python lstm_train.py --lb 10 --lr 0.001 --ep 50 --un 100


python lstm_train.py --lb 24 --lr 0.01 --ep 50 --un 25
python lstm_train.py --lb 24 --lr 0.001 --ep 50 --un 25
python lstm_train.py --lb 24 --lr 0.01 --ep 50 --un 50
python lstm_train.py --lb 24 --lr 0.001 --ep 50 --un 50
python lstm_train.py --lb 24 --lr 0.01 --ep 50 --un 100
python lstm_train.py --lb 24 --lr 0.001 --ep 50 --un 100


python lstm_train.py --lb 48 --lr 0.01 --ep 50 --un 25
python lstm_train.py --lb 48 --lr 0.001 --ep 50 --un 25
python lstm_train.py --lb 48 --lr 0.01 --ep 50 --un 50
python lstm_train.py --lb 48 --lr 0.001 --ep 50 --un 50
python lstm_train.py --lb 48 --lr 0.01 --ep 50 --un 100
python lstm_train.py --lb 48 --lr 0.001 --ep 50 --un 100

