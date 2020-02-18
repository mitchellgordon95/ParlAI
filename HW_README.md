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
ngrok http 80
copy the url
paste into lambda_function.py NGROK variable

Make an alexa app
Upload interaction_model.json and lambda_function.py # TODO: finish lambda_function.py

Open test console
> "open mitchell's sandbox app"
> "tell hey what's up"

Converse.
