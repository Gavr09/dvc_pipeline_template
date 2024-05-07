This is an example of simple DVC-pipeline which was created based on official documentation: https://dvc.org/doc/user-guide/pipelines/running-pipelines

The pipeline contains training procedure for the simple model of linear regression (sklearn realization).

The pipeline is divided into three steps:
* prepare_data - splitting initial dataset to train/test parts
* train_model - model's training on train dataset
* evaluation - evaluation of the model's metrics on test dataset
