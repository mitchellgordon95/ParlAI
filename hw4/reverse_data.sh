mkdir data_reversed
mkdir data_reversed/OpenSubtitles2018
for file in INFO LICENSE README train.txt.lengths; do
    cp data/OpenSubtitles2018/$file data_reversed/OpenSubtitles2018
done
ln -s data/OpenSubtitles2018/OpenSubtitles data_reversed/OpenSubtitles2018/OpenSubtitles

for file in train test valid; do
    cat data/OpenSubtitles2018/$file.txt | python hw4/reverse_data.py > data_reversed/OpenSubtitles2018/$file.txt
done
