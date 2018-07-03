#!/bin/bash
# Arg1: Number of folders to check for images
# Arg2: Name of approach to use when logging results
# Arg3: Number of iteration used per set of images (Ex: 10k, 20k, 30k, 40k, 50k iterations have Arg1=5, Arg3=10000)
# Arg4: Directory containing the images of the dataset

current_iter=$3
for i in $(seq 1 $1); do
	python fid.py "/home/alexia/Output/Extra/${i}" "${4}" -i "/home/alexia/Inception" --gpu "0" --output_name "${2}" --at $current_iter --output_dir "/home/alexia/Output/Extra"
	current_iter=$((current_iter + $3))
done