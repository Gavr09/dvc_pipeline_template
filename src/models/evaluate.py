import pandas as pd
import json
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import yaml
import sys
import os

def evaluate(data_path, model_path, output_dir):
    X_test_path = os.path.join(data_path, 'x_test.csv')
    y_test_path = os.path.join(data_path, 'y_test.csv')
    metrics_json = os.path.join(output_dir, 'scores.json')

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


def main():

    # get data_path and model_path from sys.argv
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file\n")
        sys.exit(1)

    data_path = sys.argv[1]
    model_dir = sys.argv[2]

    if not os.path.exists(data_path):
        sys.stderr.write("data_path is not exist!\n")
        sys.exit(1)

    if not os.path.exists(model_dir):
        sys.stderr.write("model_dir is not exist!\n")
        sys.exit(1)

    model_path = os.path.join(model_dir, 'model.pkl')

    # path for outputs
    output_dir = './reports/'
    os.makedirs(output_dir, exist_ok=True)

    evaluate(data_path, model_path, output_dir)


if __name__ == "__main__":
    main()