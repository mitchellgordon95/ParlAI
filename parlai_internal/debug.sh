#!/bin/bash
#$ -j yes
#$ -N train_dialog
#$ -o /home/hltcoe/estengel/ParlAI/parlai_internal/logs/train_dialog_lowlr.log
#$ -l 'mem_free=50G,h_rt=36:00:00,gpu=1'
#$ -q gpu.q
#$ -m ae -M elias@jhu.edu


cd /home/hltcoe/estengel/ParlAI/
source activate dialog

python -u examples/train_model.py \
        -t dailydialog  \
        -mf parlai_internal/zoo/test/test.ckpt \
        -df /tmp/dailydialog.dict \
        -bs 32 \
        -m seq2seq \
        -nt 1 \
        --hiddensize 32 \
        -vtim 3600 \
        -veps 1 \
        -sval True \
        -eps 10 \
        -vmt accuracy \
        -vmm max \
        --no-cuda 
        #-lr 0.1 \
