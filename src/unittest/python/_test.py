import pytest
from torch.utils.data import DataLoader
from pathlib import Path

import nntemplate as mynn

train_dir = 'C:\\Users\\timur\\PycharmProjects\\NNGUI\\src\\file_samples\\simpsons\\simpsons_dataset'
test_dir = 'C:\\Users\\timur\\PycharmProjects\\NNGUI\\src\\file_samples\\simpsons\\kaggle_simpson_testset\\kaggle_simpson_testset'

mynn.train_set = mynn.TrainSet(train_dir)
mynn.test_set = mynn.ValSet(train_dir, test_dir)
train_dl = DataLoader(mynn.train_set, batch_size=mynn.BS, shuffle=True)
val_dl = DataLoader(mynn.test_set, batch_size=mynn.BS * 2)


def test_tr_dataset():
    assert mynn.train_set.train_dir == Path(train_dir)


def test_ts_dataset():
    assert mynn.test_set.test_dir == Path(test_dir)


def test_len_tr_dataset():
    assert len(mynn.train_set) == 20933


# def test_len_ts_dataset():
#     assert len(mynn.test_set) == 990
