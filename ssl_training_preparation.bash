mkdir ssl-relation-prediction/src/src_data/

for dataset in FB15k-237 FB15k NELL; do
  mkdir ssl-relation-prediction/src/src_data/$dataset
  for file in train valid test; do
    echo cp data/$dataset-betae/$file.txt ssl-relation-prediction/src/src_data/$dataset/$file
    cp data/$dataset-betae/$file.txt ssl-relation-prediction/src/src_data/$dataset/$file
  done
done

cd ssl-relation-prediction/src
python preprocess_datasets.py
