import pandas as pd
import mlflow

## NOTE: You can use Microsoft Azure Machine Learning Studio for experiment tracking. Follow assignment description and uncomment below for that (you might also need to pip azureml (pip install azureml-core):
from azureml.core import Workspace
#ws = Workspace.from_config()
#ws = Workspace.from_config()
#ws = Workspace(subscription_id = "635b8853-8742-4156-907d-5f83ad2ada58", resource_group="LSDA_group", workspace_name="LSDAML", auth=None, _location=None, _disable_service_check=False, _workspace_id=None, sku='basic', tags=None, _cloud='AzureCloud')
#mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
from azureml.mlflow import register_model
experiment_name = 'experiment_with_mlflow'

def get_azureml_mlflow_tracking_uri(region, subscription_id, resource_group, workspace):
    return "azureml://{}.api.azureml.ms/mlflow/v1.0/subscriptions/{}/resourceGroups/{}/providers/Microsoft.MachineLearningServices/workspaces/{}".format(region, subscription_id, resource_group, workspace)

region='Japan East' ## example: westus
subscription_id = '635b8853-8742-4156-907d-5f83ad2ada58' ## example: 11111111-1111-1111-1111-111111111111
resource_group = 'LSDA_group' ## example: myresourcegroup
workspace = 'LSDAML' ## example: myworkspacename

MLFLOW_TRACKING_URI = get_azureml_mlflow_tracking_uri(region, subscription_id, resource_group, workspace)

## Set the MLFLOW TRACKING URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

## NOTE: Optionally, you can use the public tracking server.  Do not use it for data you cannot afford to lose. See note in assignment text. If you leave this line as a comment, mlflow will save the runs to your local filesystem.

# mlflow.set_tracking_uri("http://training.itu.dk:5000/")


{
    "subscription_id": "635b8853-8742-4156-907d-5f83ad2ada58",
    "resource_group": "LSDA_group",
    "workspace_name": "LSDAML"
}

# TODO: Set the experiment name
mlflow.set_experiment(experiment_name)

# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


# Start a run
# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
with mlflow.start_run(run_name="Test"):
    # TODO: Insert path to dataset
    df = pd.read_json("./dataset.json", orient="split")

    # TODO: Handle missing data
    class MissingValues(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self # nothing else to do
        def transform(self, X, y=None):
            df = pd.DataFrame(X)
            df.dropna(inplace=True)
            return df

    class DirectionsToInts(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self # nothing else to do
        def transform(self, X, y=None):
            df = pd.DataFrame(X)
            le = LabelEncoder()
            directionsColumn = df["Direction"]
            df["Direction"] = le.fit_transform(directionsColumn)
            return df

    pipeline = Pipeline(steps=[('MissingValues',MissingValues()),
                       ('AddWindDirections',DirectionsToInts()),
                       ('scaler', MinMaxScaler()),
                       ('RegModel', RandomForestRegressor())])

    # TODO: Currently the only metric is MAE. You should add more. What other metrics could you use? Why?
    metrics = [
        ("MAE", mean_absolute_error, []),
    ]

    X = df[["Speed","Direction"]]
    y = df["Total"]

    number_of_splits = 5

    #TODO: Log your parameters. What parameters are important to log?
    #HINT: You can get access to the transformers in your pipeline using `pipeline.steps`
    
    for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
        pipeline.fit(X.iloc[train],y.iloc[train])
        predictions = pipeline.predict(X.iloc[tetexst])
        truth = y.iloc[test]

        from matplotlib import pyplot as plt 
        plt.plot(truth.index, truth.values, label="Truth")
        plt.plot(truth.index, predictions, label="Predictions")
        plt.show()
        
        # Calculate and save the metrics for this fold
        for name, func, scores in metrics:
            score = func(truth, predictions)
            scores.append(score)
    
    # Log a summary of the metrics
    for name, _, scores in metrics:
            # NOTE: Here we just log the mean of the scores. 
            # Are there other summarizations that could be interesting?
            mean_score = sum(scores)/number_of_splits
            mlflow.log_metric(f"mean_{name}", mean_score)
            
