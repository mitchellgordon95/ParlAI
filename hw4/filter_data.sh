#! /bin/bash
#$ -cwd
#$ -V
#$ -l h_rt=1:00:00,num_proc=4,mem_free=4G
#$ -m ase
#$ -M mitchell.gordon95@gmail.com
#$ -q gpu.q@@2080

python hw4/filter_data.py data/OpenSubtitles2018/train.txt data/OpenSubtitles2018/train_filtered.txt
