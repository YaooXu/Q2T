cd ssl-relation-prediction/src
CUDA_VISIBLE_DEVICES=8 python main.py --dataset FB15k-237 --model DistMult --rank 2000 --max_epochs 200 --score_rel True \
--w_rel 1  --learning_rate 0.1 --batch_size 1000 --lmbda 0.05

CUDA_VISIBLE_DEVICES=9 python main.py --dataset FB15k-237 --model ComplEx --rank 1000 --max_epochs 200 --score_rel True \
--w_rel 1  --learning_rate 0.1 --batch_size 1000 --lmbda 0.05

cd ssl-relation-prediction/src
CUDA_VISIBLE_DEVICES=8 python main.py --dataset FB15k-237 --model CP --rank 2000 --max_epochs 200 --score_rel True \
--w_rel 1  --learning_rate 0.1 --batch_size 1000 --lmbda 0.05

cd ssl-relation-prediction/src
CUDA_VISIBLE_DEVICES=7 python main.py --dataset FB15k-237 --model TuckER --rank 512 --rank_r 512  --max_epochs 200 --score_rel True \
--w_rel 1  --learning_rate 0.1 --batch_size 1000 --lmbda 0.5