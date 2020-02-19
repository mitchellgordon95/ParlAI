conda env create --name parlai
conda activate parlai
pip install torch
pip install -r requirements.txt
python setup.py develop

scp hlt4:/home/hltcoe/estengel/ParlAI/parlai_internal/zoo/dailydialog/model2.* ParlAI/parlai_internal/zoo/dailydialog/
scp hlt4:/home/hltcoe/estengel/ParlAI/parlai_internal/dailydialog.dict ParlAI/parlai_internal/
./webserver.sh

in a separate terminal

brew cask install ngrok
ngrok tcp 80
copy the url (without leading tcp://), port
paste into lambda_function.py NGROK and NGROK_PORT variable

Make an alexa app
Upload interaction_model.json and lambda_function.py

Open test console
> "open mitchell's sandbox app"
> "tell hey what's up"

Converse. Every utterance must start with "tell".
