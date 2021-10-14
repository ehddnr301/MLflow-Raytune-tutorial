# MLFlow & ray tune

## Env setting

- `pipenv --python 3.8.8`
- `pipenv shell`
- `pip install -r requirements.txt`

## test postgresql with docker

- `docker run --rm -P -p 127.0.0.1:5431:5432 -e POSTGRES_PASSWORD=0000 -e POSTGRES_USER=ehddnr --name pgtest postgres:13.3`

## mlflow dashboard

- `mlflow server --backend-store-uri postgresql://ehddnr:0000@localhost:5431/ehddnr --default-artifact-root <실험저장 경로>`
    - `<실험저장경로>` example : `C:\Users\TFG5076XG\Documents\ray_mlflow\mlruns`

## execute

- `python test.py`