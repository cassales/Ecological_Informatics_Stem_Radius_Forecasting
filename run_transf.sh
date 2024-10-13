#!/bin/bash

# 5min tr
# Your bash script code here
# python modul_train_Transformer.py --lb 10 --ff 8 --lr 1e-6 --ep 50
# python modul_train_Transformer.py --lb 10 --ff 8 --lr 1e-7 --ep 50
# python modul_train_Transformer.py --lb 10 --ff 16 --lr 1e-6 --ep 50
# python modul_train_Transformer.py --lb 10 --ff 16 --lr 1e-7 --ep 50
# python modul_train_Transformer.py --lb 10 --ff 32 --lr 1e-6 --ep 50
# python modul_train_Transformer.py --lb 10 --ff 32 --lr 1e-7 --ep 50

# python modul_train_Transformer.py --lb 24 --ff 8 --lr 1e-6 --ep 50
# python modul_train_Transformer.py --lb 24 --ff 8 --lr 1e-7 --ep 50
# python modul_train_Transformer.py --lb 24 --ff 16 --lr 1e-6 --ep 50
# python modul_train_Transformer.py --lb 24 --ff 16 --lr 1e-7 --ep 50
# python modul_train_Transformer.py --lb 24 --ff 32 --lr 1e-6 --ep 50
# python modul_train_Transformer.py --lb 24 --ff 32 --lr 1e-7 --ep 50

# python modul_train_Transformer.py --lb 48 --ff 8 --lr 1e-6 --ep 50
# python modul_train_Transformer.py --lb 48 --ff 8 --lr 1e-7 --ep 50
# python modul_train_Transformer.py --lb 48 --ff 16 --lr 1e-6 --ep 50
# python modul_train_Transformer.py --lb 48 --ff 16 --lr 1e-7 --ep 50
# python modul_train_Transformer.py --lb 48 --ff 32 --lr 1e-6 --ep 50
# python modul_train_Transformer.py --lb 48 --ff 32 --lr 1e-7 --ep 50

# Your bash script code here
for tr in 360 720; do
    # Your commands here, using $tr
    echo "Processing with tr=$tr"
    python modul_train_Transformer.py --lb 10 --lr 1e-6 --ep 200 --ff 8 --tr $tr
    python modul_train_Transformer.py --lb 10 --lr 1e-7 --ep 200 --ff 8 --tr $tr
    python modul_train_Transformer.py --lb 10 --lr 1e-6 --ep 200 --ff 16 --tr $tr
    python modul_train_Transformer.py --lb 10 --lr 1e-7 --ep 200 --ff 16 --tr $tr
    python modul_train_Transformer.py --lb 10 --lr 1e-6 --ep 200 --ff 32 --tr $tr
    python modul_train_Transformer.py --lb 10 --lr 1e-7 --ep 200 --ff 32 --tr $tr

    python modul_train_Transformer.py --lb 24 --lr 1e-6 --ep 200 --ff 8 --tr $tr
    python modul_train_Transformer.py --lb 24 --lr 1e-7 --ep 200 --ff 8 --tr $tr
    python modul_train_Transformer.py --lb 24 --lr 1e-6 --ep 200 --ff 16 --tr $tr
    python modul_train_Transformer.py --lb 24 --lr 1e-7 --ep 200 --ff 16 --tr $tr
    python modul_train_Transformer.py --lb 24 --lr 1e-6 --ep 200 --ff 32 --tr $tr
    python modul_train_Transformer.py --lb 24 --lr 1e-7 --ep 200 --ff 32 --tr $tr

    # python modul_train_Transformer.py --lb 48 --lr 1e-6 --ep 200 --ff 8 --tr $tr
    # python modul_train_Transformer.py --lb 48 --lr 1e-7 --ep 200 --ff 8 --tr $tr
    # python modul_train_Transformer.py --lb 48 --lr 1e-6 --ep 200 --ff 16 --tr $tr
    # python modul_train_Transformer.py --lb 48 --lr 1e-7 --ep 200 --ff 16 --tr $tr
    # python modul_train_Transformer.py --lb 48 --lr 1e-6 --ep 200 --ff 32 --tr $tr
    # python modul_train_Transformer.py --lb 48 --lr 1e-7 --ep 200 --ff 32 --tr $tr
done

python modul_train_Transformer.py --lb 10 --lr 1e-6 --ep 100 --ff 8 --tr 180
python modul_train_Transformer.py --lb 10 --lr 1e-7 --ep 100 --ff 8 --tr 180
python modul_train_Transformer.py --lb 10 --lr 1e-6 --ep 100 --ff 16 --tr 180
python modul_train_Transformer.py --lb 10 --lr 1e-7 --ep 100 --ff 16 --tr 180
python modul_train_Transformer.py --lb 10 --lr 1e-6 --ep 100 --ff 32 --tr 180
python modul_train_Transformer.py --lb 10 --lr 1e-7 --ep 100 --ff 32 --tr 180

python modul_train_Transformer.py --lb 24 --lr 1e-6 --ep 100 --ff 8 --tr 180
python modul_train_Transformer.py --lb 24 --lr 1e-7 --ep 100 --ff 8 --tr 180
python modul_train_Transformer.py --lb 24 --lr 1e-6 --ep 100 --ff 16 --tr 180
python modul_train_Transformer.py --lb 24 --lr 1e-7 --ep 100 --ff 16 --tr 180
python modul_train_Transformer.py --lb 24 --lr 1e-6 --ep 100 --ff 32 --tr 180
python modul_train_Transformer.py --lb 24 --lr 1e-7 --ep 100 --ff 32 --tr 180

# python modul_train_Transformer.py --lb 48 --lr 1e-6 --ep 100 --ff 8 --tr 180
# python modul_train_Transformer.py --lb 48 --lr 1e-7 --ep 100 --ff 8 --tr 180
# python modul_train_Transformer.py --lb 48 --lr 1e-6 --ep 100 --ff 16 --tr 180
# python modul_train_Transformer.py --lb 48 --lr 1e-7 --ep 100 --ff 16 --tr 180
# python modul_train_Transformer.py --lb 48 --lr 1e-6 --ep 100 --ff 32 --tr 180
# python modul_train_Transformer.py --lb 48 --lr 1e-7 --ep 100 --ff 32 --tr 180

for tr in 30 60; do
    # Your commands here, using $tr
    echo "Processing with tr=$tr"
    python modul_train_Transformer.py --lb 10 --lr 1e-6 --ep 50 --ff 8 --tr $tr
    python modul_train_Transformer.py --lb 10 --lr 1e-7 --ep 50 --ff 8 --tr $tr
    python modul_train_Transformer.py --lb 10 --lr 1e-6 --ep 50 --ff 16 --tr $tr
    python modul_train_Transformer.py --lb 10 --lr 1e-7 --ep 50 --ff 16 --tr $tr
    python modul_train_Transformer.py --lb 10 --lr 1e-6 --ep 50 --ff 32 --tr $tr
    python modul_train_Transformer.py --lb 10 --lr 1e-7 --ep 50 --ff 32 --tr $tr

    python modul_train_Transformer.py --lb 24 --lr 1e-6 --ep 50 --ff 8 --tr $tr
    python modul_train_Transformer.py --lb 24 --lr 1e-7 --ep 50 --ff 8 --tr $tr
    python modul_train_Transformer.py --lb 24 --lr 1e-6 --ep 50 --ff 16 --tr $tr
    python modul_train_Transformer.py --lb 24 --lr 1e-7 --ep 50 --ff 16 --tr $tr
    python modul_train_Transformer.py --lb 24 --lr 1e-6 --ep 50 --ff 32 --tr $tr
    python modul_train_Transformer.py --lb 24 --lr 1e-7 --ep 50 --ff 32 --tr $tr

    # python modul_train_Transformer.py --lb 48 --lr 1e-6 --ep 50 --ff 8 --tr $tr
    # python modul_train_Transformer.py --lb 48 --lr 1e-7 --ep 50 --ff 8 --tr $tr
    # python modul_train_Transformer.py --lb 48 --lr 1e-6 --ep 50 --ff 16 --tr $tr
    # python modul_train_Transformer.py --lb 48 --lr 1e-7 --ep 50 --ff 16 --tr $tr
    # python modul_train_Transformer.py --lb 48 --lr 1e-6 --ep 50 --ff 32 --tr $tr
    # python modul_train_Transformer.py --lb 48 --lr 1e-7 --ep 50 --ff 32 --tr $tr
done