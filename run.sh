#!/bin/bash

# Your bash script code here
#python train_opt.py --lb 10 --lr 0.01 --small
#python train_opt.py --lb 24 --lr 0.01 --small
#python train_opt.py --lb 48 --lr 0.01 --small

python train_opt.py --lb 10 --lr 0.001 --small
python train_opt.py --lb 24 --lr 0.001 --small
python train_opt.py --lb 48 --lr 0.001 --small

python train_opt.py --lb 10 --lr 0.0001 --small
python train_opt.py --lb 24 --lr 0.0001 --small
python train_opt.py --lb 48 --lr 0.0001 --small

#python train_opt.py --lb 10 --lr 0.01
#python train_opt.py --lb 24 --lr 0.01
#python train_opt.py --lb 48 --lr 0.01 

python train_opt.py --lb 10 --lr 0.001
python train_opt.py --lb 24 --lr 0.001
python train_opt.py --lb 48 --lr 0.001 

python train_opt.py --lb 10 --lr 0.0001
python train_opt.py --lb 24 --lr 0.0001
python train_opt.py --lb 48 --lr 0.0001 
