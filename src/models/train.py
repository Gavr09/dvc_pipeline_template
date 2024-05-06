import pandas as pd
from sklearn.linear_model import Ridge
import pickle

X_train_path = '../../data/processed/x_train.csv'
y_train_path = '../../data/processed/y_train.csv'
ALPHA = 0.5
model_path = '../../models/model.pkl'

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)['target'].values

model = Ridge(alpha=ALPHA)
model.fit(X_train, y_train)

pickle.dump(model, open(model_path, 'wb'))