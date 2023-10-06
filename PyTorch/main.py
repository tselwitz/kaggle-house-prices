from NeuralNetwork import NeuralNetwork
from Trainer import Trainer
import pandas as pd
import torch
from torch import nn
from DataPrep import clean, split_df

def test(X, y):
    model.eval()
    with torch.inference_mode():
        print(sum(abs(model(X).T - y).T) / len(y))
        
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_filename = "data/train.csv"
    df = pd.read_csv(train_filename)
    df = clean(df)
    X, y = split_df(df, "SalePrice")
    
    train_split = int(len(X) * .8)
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    # hf = int(X_train.shape[0] / (2 * X_train.shape[1]))
    # print(hf)
    model = NeuralNetwork(X_train.shape[1], 1)
    model.to(device)
    trainer = Trainer(
        epochs=10 ** 4,
        model=model,
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=.02)
    )
    trainer.train(
        y_train,
        X_train,
        y_test,
        X_test,
        1000
    )

    test(X_train, y_train)
    test(X_test, y_test)