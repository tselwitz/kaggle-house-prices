# Kaggle House Prices
This project uses Python3 with PyTorch and XGBoost to predict housing prices from a data sample provided by Kaggle using machine learning techniques.
## Setup
Activate the virtual environment:
```
$ source .env/bin/activate
```
Install dependencies:
```
$ pip install -r req.txt
```
## Usage:
To run the PyTorch/Neural Network method to predict the housing prices, navigate to the PyTorch directory, and run the main.py
```
$ cd PyTorch
$ python3 main.py
```
Similarly, for the XGBoost method, navigate to the XGBoost directory and run the XGBoost main.py.
```
$ cd XGBoost
$ python3 main.py
```
Note: Neither method saves their results, and they will train each time they are run.