# Fune-tuning-BERT

# What is this?
This is a fine-tuning BERT project, which fine-tune BERT and trained on the custom dataset.

# How is the dataset?
the dataset contains six(or five) columns. They are respectively: the word, the POS tag of this word, the BIO tag of this word, the position of this word in the sentence, the number of sentences, and the LABEL.

# How to run this project?
Just run the extractor.py. This file will start to fine-tune the BERT and predict the test set. The output is in the result folder.

Note: Please use GPU to run this python. To use GPU, please make sure you installed the PyTorch GPU version, and have the configuration of CUDA properly.
