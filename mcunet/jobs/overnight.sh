#!/bin/bash

#python train_kvasir.py --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-112" --seed 1000 --lr 0.3
#python train_kvasir.py --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-112" --seed 2000 --lr 0.3
#python train_kvasir.py --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-112" --seed 3000 --lr 0.3

#python train_kvasir.py --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-112" --seed 1000 --lr 0.2
#python train_kvasir.py --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-112" --seed 2000 --lr 0.2
#python train_kvasir.py --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-112" --seed 3000 --lr 0.2

#python train_kvasir.py --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-112" --seed 1000 --lr 0.1

#python train_kvasir.py --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-112" --seed 1000 --lr 0.05

#python train_kvasir.py --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-112" --seed 1000 --lr 0.01

python train_kvasir.py --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-224" --resolution 224 --seed 1000 --layers 3 --lr 0.01
python train_kvasir.py --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-224" --resolution 224 --seed 1000 --layers 1 --lr 0.02

