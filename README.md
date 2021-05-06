# ASCC

Pytorch implementation of ICLR 2021 spotlight paper, "Towards Robustness Against Natural Language Word Substitutions"(https://openreview.net/forum?id=ks5nebunVn_).
Our code is partly based on https://github.com/robinjia/certified-word-sub and https://github.com/JHL-HUST/PWWS .
 
# Environment
conda env create --name ascc --file environment.yml

# Data

- Spacy
python -m spacy download en

- GloVe
cd glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
cd ..

- IMDB
cd data_set
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xvzf aclImdb_v1.tar.gz
rm -f aclImdb_v1.tar.gz
cd ..

- SNLI
cd data_set
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip
rm -r snli_1.0.zip snli_1.0 __MACOSX
cd ..

# Run

See train_imdb.sh
