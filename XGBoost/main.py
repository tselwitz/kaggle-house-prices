import xgboost
import pandas as pd

df = pd.read_csv("data/train.csv")

y = df.pop("SalePrice")
X = df

for col in X:
  if X[col].dtype == "object":
    X[col] = X[col].astype("category")

train_split = int(len(X) * .8)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

model = xgboost.XGBRegressor(enable_categorical=True, verbosity=0).fit(X_train, y_train)
preds_train = model.predict(X_train)
preds_test = model.predict(X_test)
print("\nTrain avg. error: ",abs(sum(preds_train - y_train) / len(y_train)))
print("Test avg. error: ", abs(sum(preds_test - y_test) / len(y_test)))