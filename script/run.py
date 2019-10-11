import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from script.trainDataset import trainingDataset


class myNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def main():
    torch.manual_seed(0)
    BATCH_SIZE = 4

    train = pd.read_csv('../input/train.csv', nrows=10)
    X = train.iloc[:, 1:].values
    y = train.iloc[:, 0].values
    X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                          y,
                                                          train_size=0.8,
                                                          shuffle=True,
                                                          random_state=0)

    train_dataset = trainingDataset(X_train, y_train)
    valid_dataset = trainingDataset(X_valid, y_valid)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


if __name__ == '__main__':
    main()
