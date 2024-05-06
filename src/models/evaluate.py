import pandas as pd
import json
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

X_test_path = '../../data/processed/x_test.csv'
y_test_path = '../../data/processed/y_test.csv'
metrics_json = '../../reports/scores.json'
model_path = '../../models/model.pkl'

X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)['target'].values

with open(model_path,'rb') as fp:
    model = pickle.load(fp)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

metrics_dict = {'mae':mae,
                'mse':mse,
                'rmse':rmse,
                'r2':r2}

with open(metrics_json, 'w') as fp:
    json.dump(metrics_dict, fp)