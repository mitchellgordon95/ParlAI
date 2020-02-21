# Setup 

conda env create --name parlai
conda activate parlai
pip install torch
pip install -r requirements.txt
python setup.py develop

# Dataset 
We used the (DailyDialogue dataset)[https://www.aclweb.org/anthology/I17-1099/] because it is multi-turn, lending itself nicely to the HRED setup. The dataset has 13,118 dialogues with an average of 7.9 turns per speaker per dialogue. 
The modified dataloader for this is in `parlai_internal/tasks/dailydialog/`. The dataloader returns triples of  utterances (where possible) instead of pairs.

# Model 
We implemented the HRED model in `parlai_internal/agents/hred/hred.py`. The majority of the lifting is done by the `batch_act` function and the `forward()` function in `hred_model.py`, which iterates the encoder and decoder twice, passing the context vector between them. At inference time, `eval_step()` does the same, conditioning on the history until the interaction is over. 

# Training
We tried training a number of models under various settings in an attempt to get the model to output more than one reponse. We varied the learning rate, the size of the model, the amount of time we trained for, and whether we used pre-trained word embeddings or not.

# Results 
For all of our models, the training regime has led to slightly different local optima which the model is unable to get out of. These results are reported here:

- big_model.ckpt: "I ' ll be a lot of ."
- low_lr.ckpt: "I ' m a so ." 
- model2.ckpt: "I ' ll be a good idea ." 
- glove.ckpt: "' ' ' . . . . . . . . . ." 

The only model that varies its responses is the longer training time setting: 
- long_train.ckpt: 

```
Enter Your Message: hello my name is elias
[text_1]: I ' d like to have a good of my .
Enter Your Message: hi how are you
[text_1]: I ' ll be a lot of my .
Enter Your Message: hi are you a robot
[text_1]: I ' ll be a lot of you .
Enter Your Message: where are you
[text_1]: I ' ll be a lot of my .
Enter Your Message: thanks for the info
[text_1]: That ' s right .
```

# Train/valid loss graphs
Train/valid loss graphs are in `parlai_internal/zoo/dailydialog/*.png`

# Alexa integration 
Copy the models: current models are stored on the COE grid: 

```
conda activate parlai

scp hlt4:/home/hltcoe/estengel/ParlAI/parlai_internal/zoo/dailydialog/model2.* ParlAI/parlai_internal/zoo/dailydialog/
scp hlt4:/home/hltcoe/estengel/ParlAI/parlai_internal/dailydialog.dict ParlAI/parlai_internal/
./webserver.sh
```

in a separate terminal

```
brew cask install ngrok
ngrok tcp 80
``` 

copy the url (without leading tcp://), port
paste into lambda_function.py NGROK and NGROK_PORT variable

Make an alexa app
Upload interaction_model.json and lambda_function.py

Open test console
> "open mitchell's sandbox app"
> "tell hey what's up"

Converse. Every utterance must start with "tell".
