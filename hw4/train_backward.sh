#! /bin/bash
#$ -cwd
#$ -V
#$ -l gpu=1,h_rt=48:00:00,num_proc=2,mem_free=3G
#$ -m ase
#$ -M mitchell.gordon95@gmail.com
#$ -q gpu.q@@2080

ml load cuda10.0/toolkit
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi

python -m parlai.scripts.train_model \
	     -t opensubtitles \
	     -mf parlai_internal/backward.ckpt \
	     -bs 16 \
	     -m transformer/generator \
       --dict-maxtokens 30000 \
	     -stim 1800 \
	     -sval True \
	     --optimizer adam \
	     -lr .01 \
	     --embedding-type fasttext_cc \
	     --beam-size 5 \
	     --dropout 0.1 \
	     --attention-dropout 0.1 \
	     -eps 10 \
	     -ttim 86400 \
	     -lfc True \
	     --truncate 1024 \
       -dp data_reversed
