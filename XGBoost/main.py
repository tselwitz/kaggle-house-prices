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

def show_results(pred, y):
  avg_real = sum(y) / len(y)
  avg_pred = sum(pred) / len(pred)
  avg_deviation = sum(abs(pred.T - y).T) / len(y)
  print(f"Avg. Price (Real): ${avg_real:.2f}")
  print(f"Avg. Price (Predicted): ${avg_pred:.2f}")
  print(f"Avg. Deviation: ${avg_deviation:.2f}")
  print(f"% Accuracy (Avg. Deviation / Avg. Real): {1 - (avg_deviation / avg_real):.2%}")

print("\nTrain:")
show_results(preds_train, y_train)
print("\nTest:")
show_results(preds_test, y_test)