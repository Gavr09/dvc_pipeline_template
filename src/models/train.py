import pandas as pd
from sklearn.linear_model import Ridge
import pickle
import yaml
import sys
import os

def train_model(data_path, alpha, output_dir):
    X_train_path = os.path.join(data_path, 'x_train.csv')
    y_train_path = os.path.join(data_path, 'y_train.csv')
    model_path = os.path.join(output_dir, 'model.pkl')

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)['target'].values

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    pickle.dump(model, open(model_path, 'wb'))

def main():

    # read params from yaml-file
    params = yaml.safe_load(open("params.yaml"))["train_model"]
    alpha = params["alpha"]

    # get data_path from sys.argv
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file\n")
        sys.exit(1)

    data_path = sys.argv[1]

    if not os.path.exists(data_path):
        sys.stderr.write("PATH_TRAIN is not exist!\n")
        sys.exit(1)

    # path for outputs
    output_dir = './models/'
    os.makedirs(output_dir, exist_ok=True)

    train_model(data_path, alpha, output_dir)


if __name__ == "__main__":
    main()
