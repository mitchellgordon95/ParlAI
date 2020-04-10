mkfifo piper
sudo nc -l 80 -k < piper | python examples/interactive.py -mf parlai_internal/zoo/dailydialog/model2.ckpt > piper

