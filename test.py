import mlflow
import pandas as pd
import xgboost as xgb
import random
from sklearn.linear_model import LogisticRegression

from mlflow.tracking import MlflowClient
from abc import ABC, abstractmethod
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback, mlflow_mixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from db import engine

SELECT_ALL_INSURANCE = """
        SELECT *
        FROM insurance
    """

HOST_URL = 'http://localhost:5001'

EXP_NAME = 'my_experiment143'

METRIC = 'mae'

class ETL:
    def __init__(self):
        self.df = None

    def _extract(self, data_extract_query):
        self.df = pd.read_sql(data_extract_query, engine)

    def _scaling(self, scale_list, scaler):
        self.df.loc[:, scale_list] = scaler().fit_transform(
            self.df.loc[:, scale_list]
        )

    def _encoder(self, enc_list, encoder):
        for col in enc_list:
            self.df.loc[:, col] = encoder().fit_transform(self.df.loc[:, col])

    def _load(self):
        return self.df.iloc[:, :-1].values, self.df.iloc[:, -1].values

    def exec(self, data_extract_query, *args):
        self._extract(data_extract_query)
        if args is not None:
            for trans_list, transformer in args:
                if "encoder" in transformer.__name__.lower():
                    self._encoder(trans_list, transformer)
                elif "scaler" in transformer.__name__.lower():
                    self._scaling(trans_list, transformer)
                else:
                    break
        return self._load()

class Tuner(ABC):
    def __init__(self):
        self.model = None
        self.data_X = None
        self.data_y = None
        self.config = None

    def _split(self, test_size):
        """
        self.data_X, self.data_y 를 split
        data_X와 data_y는 상속받은 class에서 값을 받게 되어있음.
        """
        train_X, valid_X, train_y, valid_y = train_test_split(
            self.data_X,
            self.data_y,
            test_size=test_size,
        )

        return train_X, valid_X, train_y, valid_y

    @abstractmethod
    def exec(self):
        pass

class InsuranceTuner(Tuner):

    def __init__(self, data_X, data_y):
        self.data_X = data_X
        self.data_y = data_y
        self.TUNE_METRIC_DICT = {
            "mae": "min",
            "mse": "min",
            "rmse": "min"
        }

    def _log_experiments(self, config, metrics, xgb_model):
        best_score = None
        mlflow.set_tracking_uri(HOST_URL)
        mlflow.set_experiment(EXP_NAME)

        client = MlflowClient()
        exp_id = client.get_experiment_by_name(EXP_NAME).experiment_id
        runs = mlflow.search_runs([exp_id])

        if len(runs) > 0:
            try:
                best_score = runs[f'metrics.{METRIC}'].min()
            except Exception as e:
                print(e)

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_metrics(metrics)
            mlflow.log_params(config)
            
            if not best_score or best_score > metrics[METRIC]:
                print('log model')
                mlflow.xgboost.log_model(
                    xgb_model,
                    artifact_path="model",
                )

    def _trainable(self, config):
        train_x, test_x, train_y, test_y = super()._split(0.2)
        train_set = xgb.DMatrix(train_x, label=train_y)
        test_set = xgb.DMatrix(test_x, label=test_y)

        results = {}
        xgb_model = xgb.train(
            config,
            train_set,
            evals=[(test_set, "eval")],
            evals_result=results,
            verbose_eval=False
        )
        return results['eval'], xgb_model

    def _run(self, config):
        results, xgb_model = self._trainable(config)

        metrics = {
            "mae": min(results["mae"]),
            "rmse": min(results["rmse"]),
        }

        self._log_experiments(config, metrics, xgb_model)
        tune.report(**metrics)

    def exec(self, tune_config=None, num_trials=15):
        DEFAULT_CONFIG = {
            "objective": "reg:squarederror",
            "eval_metric": ["mae", "rmse"],
            "max_depth": tune.randint(1, 9),
            "min_child_weight": tune.choice([1, 2, 3]),
            "subsample": tune.uniform(0.5, 1.0),
            "eta": tune.loguniform(1e-4, 1e-1),
        }

        config = tune_config if tune_config else DEFAULT_CONFIG
        tune.run(
            self._run,
            config=config,
            metric=METRIC,
            mode=self.TUNE_METRIC_DICT[METRIC],
            num_samples=num_trials,
        )

if __name__ == '__main__':
    mlflow.set_tracking_uri(HOST_URL)
    mlflow.set_experiment(EXP_NAME)

    etl = ETL()

    trans1 = [["sex", "smoker", "region"], LabelEncoder]
    trans2 = [["age", "bmi", "children"], StandardScaler]

    X, y = etl.exec(SELECT_ALL_INSURANCE, trans1, trans2) # 전처리가 끝난 데이터

    it = InsuranceTuner(
        data_X=X,
        data_y=y
    )
    it.exec()