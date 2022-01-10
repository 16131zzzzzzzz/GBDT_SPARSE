# GBDT-SPARSE

This is a codebase for ML class. The general principle is in `report.md`.

## Requirements
- python 3
- numpy
- pandas
- tqdm

## Train your own GBDT

### Prepare data

We use data from the GBDT-SPARSE paper. You can download and find readme of it [here](http://manikvarma.org/downloads/XC/XMLRepository.html). Put it in the `data` folder.



### Write config

Examples of config file are in the `configs` folder.

```json
[
    {
        "max_depth" : 10,
        "min_points" : 100,
        "lambd" : 5,
        "sparse_k" : 20,
        "learn_rate" : 0.8,
        "sample_rate" : 0.8,
        "max_iter" : 1000,
        "early_stop_n" : 1,
        "classify_shrd" : 0.66
    }
]
```

- `max_depth` :  max depth of trees

- `min_points` : leafs stop splitting if the point number is less than it

- `lambd` : hyperparameter  for regularizer

- `sparse_k` : hyperparameter in paper

- `learn_rate` : learning rate

- `sample_rate` : percentage of training data

- `max_iter` : max number of trees

- `early_stop_n` : (in development)

- `classify_shrd` : threshold for turning probability to label

### Training and Prediction

Examples can be found in `run_{dataset}.ipynb` files.

`data_format.py` shows the format of data.

### Other files

- `{}_draw.ipynb` are plotting files

- The `record.csv` records all results of experiments

- main part of GBDT-SPARSE is in `model` folder

- the results are saved in `result` folder

## Reference

from paper

```tex
@inproceedings{si2017gradient,
  title={Gradient boosted decision trees for high dimensional sparse output},
  author={Si, Si and Zhang, Huan and Keerthi, S Sathiya and Mahajan, Dhruv and Dhillon, Inderjit S and Hsieh, Cho-Jui},
  booktitle={International conference on machine learning},
  pages={3182--3190},
  year={2017},
  organization={PMLR}
}
```
