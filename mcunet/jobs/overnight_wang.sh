#!/bin/bash

python train_kvasir.py --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-224" --resolution 224 --seed 1000 --layers 3 --lr 0.1 --batch_size 16
python train_kvasir.py --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-224" --resolution 224 --seed 1000 --layers 3 --lr 0.2 --batch_size 16
python train_kvasir.py --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-224" --resolution 224 --seed 1000 --layers 3 --lr 0.05 --batch_size 16

python train_kvasir.py --arch resnet32 --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-224" --resolution 224 --seed 1000 --layers 3 --lr 0.1 --batch_size 16
python train_kvasir.py --arch resnet32 --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-224" --resolution 224 --seed 1000 --layers 3 --lr 0.2 --batch_size 16
python train_kvasir.py --arch resnet32 --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-224" --resolution 224 --seed 1000 --layers 3 --lr 0.05 --batch_size 16

python train_kvasir.py --arch resnet44 --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-224" --resolution 224 --seed 1000 --layers 3 --lr 0.1 --batch_size 16
python train_kvasir.py --arch resnet44 --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-224" --resolution 224 --seed 1000 --layers 3 --lr 0.2 --batch_size 16
python train_kvasir.py --arch resnet44 --data_dir "C:\Users\Michael Ruan\Documents\Github\tinyml\mcunet\data\kvasir-dataset-v2-cropped-224" --resolution 224 --seed 1000 --layers 3 --lr 0.05 --batch_size 16

