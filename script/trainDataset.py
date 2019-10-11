import torch
import numpy as np
from torch.utils.data import Dataset


class trainingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].astype(np.float), self.y[idx]


class testDataset(Dataset):
    def __init__(self, df_test):
        self.df_test = df_test

    def __len__(self):
        return len(self.df_test)

    def __getitem__(self, idx):
        return self.df_test.iloc[idx, :].values.astype(np.float)


def main():
    data_num = 10
    X_train = torch.ones(size=(data_num, 784))
    y_train = torch.ones(size=(data_num,))

    train_dataset = trainingDataset(X=X_train, y=y_train)
    assert len(train_dataset) == data_num

    x, y = next(iter(train_dataset))
    assert torch.Size([784]) == x.size()
    assert torch.float == x.dtype
    assert 1.0 == y


if __name__ == '__main__':
    main()
