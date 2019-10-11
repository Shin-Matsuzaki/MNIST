import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import DataLoader

from script.myNetwork import mySimpleNet
from script.trainDataset import trainingDataset, testDataset


def main():
    torch.manual_seed(0)
    BATCH_SIZE = 160

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

    # モデル
    model: nn.Module = mySimpleNet()

    # 最適化アルゴリズムと損失関数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # 学習
    num_epochs = 2
    model.train()   # 学習モード
    for epoch in range(1, num_epochs + 1):
        for images, labels in train_loader:
            optimizer.zero_grad()   # 勾配初期化
            output = model(images.float())  # 順伝播計算
            loss = criterion(output, labels)    # 損失計算
            loss.backward()     # Backward
            optimizer.step()    # 重み更新
            print(loss.item())

    # TestDataでの予測
    df_test = pd.read_csv('../input/test.csv')
    test_dataset = testDataset(df_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()    # 評価モード
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            outputs = model(images)

            _, y_pred = torch.max(outputs, dim=1)
            y_pred_label = y_pred.numpy()
            predictions.append(y_pred_label)
            exit()


if __name__ == '__main__':
    main()
