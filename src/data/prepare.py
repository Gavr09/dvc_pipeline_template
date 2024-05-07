import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import sys
import os


def prepare_data(data_path, random_state, output_dir):

    data = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['target']), data['target'], test_size=0.33, random_state=random_state)

    X_train.to_csv(os.path.join(output_dir, 'x_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'x_test.csv'), index=False)
    y_train.to_frame().to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_frame().to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)


def main():

    # read params from yaml-file
    params = yaml.safe_load(open("params.yaml"))["prepare_data"]
    random_state = params["random_state"]

    # get data_path from sys.argv
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file\n")
        sys.exit(1)

    data_path = sys.argv[1]

    if not os.path.exists(data_path):
        sys.stderr.write("data_path is not exist!\n")
        sys.exit(1)

    # path for outputs
    output_dir = './data/processed/'
    os.makedirs(output_dir, exist_ok=True)

    prepare_data(data_path, random_state, output_dir)


if __name__ == "__main__":
    main()
