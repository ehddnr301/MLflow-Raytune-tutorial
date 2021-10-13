import mlflow
import pandas as pd
import xgboost as xgb

from abc import ABC, abstractmethod
from ray import tune
from ray.tune.integration.mlflow import mlflow_mixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from db import engine

SELECT_ALL_INSURANCE = """
        SELECT *
        FROM insurance
    """

POSTGRES_URL = 'postgresql://ehddnr:0000@localhost:5431/ehddnr'

EXP_NAME = 'my_experiment12'

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

    def __init__(self, data_X, data_y, config):
        self.data_X = data_X
        self.data_y = data_y
        self.config = config

    @mlflow_mixin
    def _train_insurance(self, config):
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

        result_mae = results['eval']['mae'][-1] # 실험 결과
        exp_id = mlflow.get_experiment_by_name(EXP_NAME).experiment_id
        run_list = mlflow.list_run_infos(exp_id)

        metric_list = []
        for run in run_list:
            mae_list = mlflow.tracking.MlflowClient().get_metric_history(run.run_id, 'mae')
            if len(mae_list) != 0:
                val = mae_list[0].value
                metric_list.append(val)

        if len(metric_list) != 0:
            minimum = min(metric_list)
        else:
            minimum = 9999999

        if result_mae <= minimum:
            params = {
                "max_depth": config['max_depth'],
                "min_child_weight": config['min_child_weight'],
                "subsample": config['subsample'],
                "eta": config['eta'],
            }
            
            mlflow.log_params(params)
            mlflow.log_metric('mae', result_mae)
            mlflow.xgboost.log_model(
                xgb_model=xgb_model,
                artifact_path=''
            )
            tune.report(mean_loss=result_mae, done=True)


    def exec(self):
        """
        exec method가 실행되면 _train_insurance method를 이용하여 tune.run 실행
        """
        tune.run(
                self._train_insurance,
                config=self.config,
                num_samples=10
            )


if __name__ == '__main__':
    mlflow.set_tracking_uri(POSTGRES_URL)
    mlflow.set_experiment(EXP_NAME)

    etl = ETL()

    trans1 = [["sex", "smoker", "region"], LabelEncoder]
    trans2 = [["age", "bmi", "children"], StandardScaler]

    X, y = etl.exec(SELECT_ALL_INSURANCE, trans1, trans2) # 전처리가 끝난 데이터

    it = InsuranceTuner(
        data_X=X,
        data_y=y,
        config={
            "objective": "reg:squarederror",
            "eval_metric": ["mae", "rmse"],
            "max_depth": tune.randint(1, 9),
            "min_child_weight": tune.choice([1, 2, 3]),
            "subsample": tune.uniform(0.5, 1.0),
            "eta": tune.loguniform(1e-4, 1e-1),
            "mlflow": {
                'experiment_name': EXP_NAME,
                'tracking_uri': mlflow.get_tracking_uri()
            }
        }
    )
    it.exec()