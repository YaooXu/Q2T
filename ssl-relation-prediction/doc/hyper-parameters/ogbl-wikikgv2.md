# Hyper-parameters for ogbl-wikikgv2
## Grid-Search
The hyper-parameters were tuned by grid-search on the validation sets. The runs with best validation MRR are selected. The base model is [ComplEx](https://www.jmlr.org/papers/volume18/16-563/16-563.pdf) with [N3 regularizer](https://arxiv.org/pdf/1806.07297.pdf). The embedding initialization is 1e-3. The optimizer is [Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html).

For ComplEx without relation prediction, we searched
```python
learning_rate=[1e-1, 1e-2],
batch_size=[100, 500],
lmbda=[0.0005, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 0],
w_rel=[0],
score_rel=[False], 
```

For ComplEx with relation prediction, we searched
```python
learning_rate=[1e-1, 1e-2],
batch_size=[100, 250, 500],
lmbda=[0.0005, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 0],
w_rel=[2, 1, 0.5, 0.25, 0.125], # weighting of the relation prediction objective
score_rel=[True], # turn on the relation prediction objective
```
We used the same grid for all the CoDEx datasets. Better performance can be found by customizing the grid for each dataset.

## Best Run for ogbl-wikikgv2, Rank=25
```python
learning_rate=[1e-1],
batch_size=[100],
lmbda=[0.05],
w_rel=[0.5], # weighting of the relation prediction objective
score_rel=[True], # turn on the relation prediction objective
```
## Best Run for ogbl-wikikgv2, Rank=50
```python
learning_rate=[1e-1],
batch_size=[250],
lmbda=[0.1],
w_rel=[0.125], # weighting of the relation prediction objective
score_rel=[True], # turn on the relation prediction objective
```

### Trained Embeddings for the Best Run
You can download the trained embeddings for the best run from [here](https://dl.fbaipublicfiles.com/ssl-relation-prediction/complex/ogbl-wikikg2.zip). Unzip the model by
```
unzip ogbl-wikikg2.zip
```
After unzipping you will get 3 files: `best_valid.model` (Pytorch model file), `ent_id` (Enitity IDs)and `rel_id` (Relation IDs). You can load the embeddings by running the following python script
```
import torch
state_dict = torch.load('best_valid.model', map_location='cpu') # load on cpu
entity_embedding = state_dict[''embeddings.0.weight'] # get entity embeddings
relation_embedding = state_dict[''embeddings.1.weight'] # get relation embeddings
``` 
Or you can load the `ComplEx` model using our codebase by running the following python scripts under `src`
```
from models import ComplEx
model = ComplEx(sizes=[2500604,1070,2500604], rank=50, init_size=1e-3)
state_dict = torch.load('best_valid.model', map_location='cpu') # load on cpu
model.load_state_dict(state_dict)
```
## Best Run for ogbl-wikikgv2, Rank=100
```python
learning_rate=[1e-1],
batch_size=[250],
lmbda=[0.05],
w_rel=[0.25], # weighting of the relation prediction objective
score_rel=[True], # turn on the relation prediction objective
```

These configurations of hyper-parameters should reproduce [the results](https://github.com/facebookresearch/ssl-relation-prediction#ogbl-wikikg2).


## Summary
 Compared to full ranking for conventional KBC datasets, the evaluation in ogbl-wikikgv2 uses a fixed sampled set of 500 entities for both (h,r,?) and (?,r,t) queries. Some training runs were preempted. Additional training time would lead to better results.