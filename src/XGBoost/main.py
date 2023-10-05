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
preds = model.predict(X_test)

print(abs(sum(preds - y_test) / len(y_test)))