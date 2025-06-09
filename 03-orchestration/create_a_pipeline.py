import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("yellow-taxi-lr-model")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


def read_dataframe(filename):
    df = pd.read_parquet(filename)

    print(f"Loaded {len(df):,} rows from {filename}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    print(f"📏 Dataset shape after filtering: {df.shape}")

    return df


def train_and_log_model(df):
    # Sample down to avoid memory overload
    df = df.sample(n=100_000, random_state=42)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    target = 'duration'
    features = ['PU_DO', 'trip_distance']

    dv = DictVectorizer()
    train_dicts = train_df[features].to_dict(orient='records')
    val_dicts = val_df[features].to_dict(orient='records')

    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    y_train = train_df[target].values
    y_val = val_df[target].values

    with mlflow.start_run() as run:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_val)

        # Compute RMSE manually
        mse = mean_squared_error(y_val, y_pred)
        rmse = mse ** 0.5

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", rmse)

        # Save and log model
        model_path = models_folder / "linear_regression.bin"
        with open(model_path, "wb") as f_out:
            pickle.dump((dv, lr), f_out)

        mlflow.log_artifact(local_path=str(model_path), artifact_path="models")

        # Stats
        model_size = os.path.getsize(model_path)
        print(f"📈 Intercept of model: {lr.intercept_:.4f}")
        print(f"💾 Model size (bytes): {model_size:,}")
        print(f"🏃 View run {run.info.run_id} at: http://localhost:5000/#/experiments/1/runs/{run.info.run_id}")

        return run.info.run_id


def run():
    year = 2023
    month = 3
    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    df = read_dataframe(filename)
    run_id = train_and_log_model(df)
    print(f"MLflow run_id: {run_id}")


if __name__ == "__main__":
    run()
