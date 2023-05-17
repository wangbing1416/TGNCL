# TGNCL

The source code of our paper "_Graph-based Text Classification by Contrastive Learning with Text-level Graph Augmentation_"

Our released code follows [VPALG: Paper-publication Prediction with Graph Neural Networks](https://dl.acm.org/doi/abs/10.1145/3459637.3482490).

### Requirement

```
torch <= 1.13.1
cudatoolkit == 11.6.1
dgl == 1.1.0.cu113
numpy == 1.23.5
```

### Train 

1. prepare and download the four datasets from their original papers to `./data`
2. download `glove.6B.300d.txt` to `./`
3. run the python file
```shell
python train.py
```
4. the model will be saved to the root `./`

### Cite
```
NONE
```