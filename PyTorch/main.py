from NeuralNetwork import NeuralNetwork
from Trainer import Trainer
import pandas as pd
import torch
from torch import nn
from DataPrep import clean, split_df

def test(X, y):
    model.eval()
    with torch.inference_mode():
        pred = model(X)
        avg_real = (sum(y) / len(y)).item()
        avg_pred = (sum(pred) / len(pred)).item()
        avg_deviation = (sum(abs(pred.T - y).T) / len(y)).item()
        print(f"Avg. Price (Real): ${avg_real:.2f}")
        print(f"Avg. Price (Predicted): ${avg_pred:.2f}")
        print(f"Avg. Deviation: ${avg_deviation:.2f}")
        print(f"% Accuracy (Avg. Deviation / Avg. Real): {1 - (avg_deviation / avg_real):.2%}")
        
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

    model = NeuralNetwork(X_train.shape[1], 1)
    model.to(device)
    trainer = Trainer(
        epochs=10 ** 2,
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

    print("\nTrain:")
    test(X_train, y_train)
    print("\nTest:")
    test(X_test, y_test)