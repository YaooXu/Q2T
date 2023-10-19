# Query2Triple: Unified Query Encoding for Answering Diverse Complex Queries over Knowledge Graphs


This repository is based on [KGReasoning](https://github.com/snap-stanford/KGReasoning), containing implementation for paper `Query2Triple: Unified Query Encoding for Answering Diverse Complex Queries over Knowledge Graphs` (https://arxiv.org/abs/2310.11246).

In this documentation, we detail how to train KGE checkpoints and use this checkpoints to reproduce out results in the paper.

## Requirement of this repository and submodule
- networkx
- numpy
- ogb
- pandas
- pytz
- scikit_learn
- scipy
- tensorboardX
- torch
- tqdm

More details can be found in `requirement.txt`.

## Preparation

### (A) Prepare the dataset

Please download the dataset from  [snap-stanford/KGReasoning](https://github.com/snap-stanford/KGReasoning).

Specifically, one can run:
```bash
mkdir data
cd data
wget http://snap.stanford.edu/betae/KG_data.zip # a zip file of 1.3G
unzip KG_data.zip
```

Then the `data` folder will contain the following folders and files:
```
FB15k-237-betae
FB15k-237-q2b
FB15k-betae
FB15k-q2b
KG_data.zip
NELL-betae
NELL-q2b
```

### (B) Pretrain KGE with ssl-relation-prediction
The directory `ssl-relation-prediction` is forked from [ssl-relation-prediction](https://github.com/facebookresearch/ssl-relation-prediction).
We make some change to this repository to make the submodule generate checkpoints that can be directly used by our model.

#### (1) Preprocess datasets

Run this script to preprocess datasets for the submodule.
```bash
bash ssl_training_preparation.bash
```

#### (2) Pretrain KGE

The commands to train ComplEx checkpoints for each datasets are as follows:
```bash
cd ssl-relation-prediction/src

# FB15k
python main.py --dataset FB15k --model ComplEx --rank 1000 --max_epochs 200 --score_rel True \
--w_rel 0.01  --learning_rate 0.1 --batch_size 1000 --lmbda 0.01

# FB15k-237
python main.py --dataset FB15k-237 --model ComplEx --rank 1000 --max_epochs 200 --score_rel True \
--w_rel 4  --learning_rate 0.1 --batch_size 1000 --lmbda 0.05

# NELL
python main.py --dataset NELL --model ComplEx --rank 1000 --max_epochs 200 --score_rel True \
--w_rel 0.1  --learning_rate 0.1 --batch_size 1000 --lmbda 0.05
```

#### (3) Train TGT
Notice: assign KGE checkpoint path to $kge_ckpt_path, such as `ssl-relation-prediction/src/ckpts/FB15k/ComplEx-2023.05.06-20_57_11/best_valid.model`. 


Sample usage at FB15k.
```bash
python main.py --cuda --do_train --do_valid --do_test  --data_path data/FB15k-betae --kge_ckpt_path $kge_ckpt_path -b 1024 -n 512 -de 2000 -dr 2000 -lr 0.0004 --label_smoothing 0.4 --cpu_num 5 --geo complex --num_hidden_layers 6 --num_attention_heads 16 --hidden_size 768 --intermediate_size 768 --token_embeddings 0 --hidden_dropout_prob 0.1 --warm_up_steps 20000 --max_steps 200000 --valid_steps 5000 --tasks 1p.2p.3p.2i.3i.ip.pi.2u.up.2in.3in.inp.pin.pni --prefix logs
```

Sample usage at FB15k-237.
```bash
python main.py --cuda --do_train --do_valid --do_test  --data_path data/FB15k-237-betae --kge_ckpt_path $kge_ckpt_path -b 1024 -n 512 -de 2000 -dr 2000 -lr 0.0004 --label_smoothing 0.6 --cpu_num 5 --geo complex --num_hidden_layers 6 --num_attention_heads 16 --hidden_size 768 --intermediate_size 768 --token_embeddings 0 --hidden_dropout_prob 0.1 --warm_up_steps 20000 --max_steps 200000 --valid_steps 5000 --tasks 1p.2p.3p.2i.3i.ip.pi.2u.up.2in.3in.inp.pin.pni --prefix logs
```

Sample usage at NELL.
```bash
python main.py --cuda --do_train --do_valid --do_test  --data_path data/NELL --kge_ckpt_path $kge_ckpt_path -b 1024 -n 512 -de 2000 -dr 2000 -lr 0.0005 --label_smoothing 0.6 --cpu_num 5 --geo complex --num_hidden_layers 6 --num_attention_heads 12 --hidden_size 768 --intermediate_size 768 --token_embeddings 0 --hidden_dropout_prob 0.1 --warm_up_steps 20000 --max_steps 200000 --valid_steps 5000 --tasks 1p.2p.3p.2i.3i.ip.pi.2u.up.2in.3in.inp.pin.pni --prefix logs
```
