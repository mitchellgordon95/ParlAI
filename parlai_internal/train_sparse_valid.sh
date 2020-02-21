#!/bin/bash
#$ -j yes
#$ -N train_dialog
#$ -o /home/hltcoe/estengel/ParlAI/parlai_internal/logs/train_sparse_valid.log
#$ -l 'mem_free=50G,h_rt=36:00:00,gpu=1'
#$ -q gpu.q
#$ -m ae -M elias@jhu.edu


cd /home/hltcoe/estengel/ParlAI/
source activate dialog

python -u examples/train_model.py \
        -t internal:dailydialog  \
        -mf parlai_internal/zoo/dailydialog/long_train.ckpt \
        -df /home/hltcoe/estengel/ParlAI/parlai_internal/dailydialog.dict \
        -bs 64 \
        -m internal:hred  \
        -nt 1 \
        --hiddensize 512 \
        --contextsize 512 \
        -vtim 10800 \
        -veps 2 \
        -sval True \
        -eps 10 \
        -vmt token_acc_2 \
        -vmm max \
