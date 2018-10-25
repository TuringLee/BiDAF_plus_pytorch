# BiDAF_plus

Reproduce [**BiDAF**](https://arxiv.org/pdf/1611.01603.pdf) based on pytorch.

Reuse the framework of [**DrQA**](https://arxiv.org/pdf/1704.00051.pdf).

Probably the best BiDAF implementation in pytorch.

The BaseLine EM = **68.41** .
And you can even get higher scores by set **Elmo**, **Slef_attn**, **lrshrink**, **tune_partial** and **add_features**.

## Requirement:
- python 3.6
- pytorch 0.4
- json
- numpy
- spacy
- logging
- argparse
- functool
- collections
- multiprecessing

## Useage:
- Install spacy and download the en model.
1. Preprocess
-  python preprocess.py path_to_SQuAD_dir path_to_save_dir --split(train/dev) --workers 1 -- tokenizer spacy
2. Training
- CUDA_VISIBLE_DEVICES=0 python train.py --save_dir(model_save_dir) --log_file(path_to_log_file) --embedding_file(path_to_embedding_file[glove.6B.100d.txt]) --train_file(path_to_preprocessed_training_file) --dev_file(path_to_preprocessed_dev_file)  --dev_json(path_to_original_dev_json) --network_file BiDAF \[--add_features --lrshrink 0.5 --tune_partial 100\]

## Todo
- Test hyperparameters combination to get the highest score.