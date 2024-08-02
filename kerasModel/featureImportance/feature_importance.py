import os
import logging
import datetime
import dask
from dask import delayed
import cupy as cp
import numpy as np
from typing import Optional, List
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from tensorflow import keras
from dataLoader.data_loader_dask import DataLoaderDask

logging.basicConfig(level=logging.INFO)

class FeatureImportanceEvaluator:
    def __init__(self, modelPath: str, dataFrame: dd.DataFrame, sampleFraction: float = 1.0, nRepeats: int = 20):
        """
        Initialize the FeatureImportanceEvaluator with a model and Dask DataFrame.

        Args:
            modelPath (str): Path to the Keras model.
            dataFrame (dd.DataFrame): Dask DataFrame containing features and target.
            sampleFraction (float): sample ratio for the data.
            nRepeats (int): Number of repeats for permutation importance evaluation.
        """
        self.model = keras.models.load_model(modelPath)
        logging.info("Model loaded successfully.")

        self.dataFrame = dataFrame.sample(frac = sampleFraction, random_state = 42)

        self.x = self.dataFrame.drop(columns = ['target'])  # Features as Dask DataFrame
        self.y = self.dataFrame['target']                 # Target as Dask Series

        self.xPandas = self.x.compute()              # Features as pandas DataFrame
        self.yPandas = self.y.compute()              # Target as pandas Series

        self.nRepeats = nRepeats
        os.makedirs('featureImportancePlots', exist_ok = True)
        logging.info("DataFrame processed and samples created.")

    def _predict(self, partition):
        """Predict classes for a partition using the Keras model."""
        predictions = self.model.predict(partition)
        predictedClasses = (predictions.flatten() > 0.5).astype(int)
        logging.debug(f"Predicted classes for a partition: {predictedClasses}")
        return predictedClasses

    def scoreFunctionWithBatches(self, X: Optional[dd.DataFrame] = None) -> float:
        """Calculate accuracy score using batch predictions and work with dask dataframe."""
        X = self.x if X is None else X
        predictedClasses = X.map_partitions(lambda part: self._predict(part.to_numpy()), meta = pd.Series(dtype='int'))
        predictedClasses = predictedClasses.compute()
        return accuracy_score(self.yPandas, predictedClasses)

    def computeFeatureImportance(self) -> pd.DataFrame:
        """Compute feature importance using permutation importance with SciKit."""

        def scoreFunction(model, X, y):
            predictions = model.predict(X)
            predictedClasses = (predictions > 0.5).astype(int)
            return accuracy_score(y, predictedClasses)

        baselineScore = scoreFunction(self.model, self.xPandas, self.yPandas)
        logging.info(f"Baseline Score (Accuracy): {baselineScore}")

        result = permutation_importance(self.model, self.xPandas, self.yPandas, n_repeats = self.nRepeats, random_state = 42, scoring = scoreFunction)

        # Create a DataFrame for importance values
        featureImportanceDataFrame = pd.DataFrame({
            'Feature': self.xPandas.columns,
            'Importance': result.importances_mean,
            'Std': result.importances_std
        }).sort_values(by = 'Importance', ascending = False)

        return featureImportanceDataFrame

    def computeFeatureImportanceDask(self) -> pd.DataFrame:
        """Compute feature importance using Dask. Returns Pandas dataframe for plotting"""
        baselineScore = self.scoreFunctionWithBatches()
        logging.info(f"Baseline Score (Accuracy): {baselineScore}")

        importanceTasks = []
        for feature in self.x.columns:      # Iterate through each feature and create delayed tasks for each permutation
            tasks = self._createPermutationTasks(feature)
            importanceTasks.extend(tasks)

        results = dask.compute(*importanceTasks)        # Compute the results in parallel
        logging.info("Permutation importance evaluations completed.")

        return self.getFeatureImportanceDataframe(results)

    def _createPermutationTasks(self, feature: str) -> List[delayed]:
        """Create delayed tasks for permuting a feature. Returns a list of delayed tasks."""
        return [delayed(self.evaluatePermutation)(feature) for _ in range(self.nRepeats)]       # Repeat nRepeats

    def evaluatePermutation(self, feature: str) -> float:
        """Evaluate the permutation of a feature and return the score."""
        XPermuted = self.x.map_partitions(lambda dataFrame: dataFrame.copy())       # Copy the Dask DataFrame
        XPermuted = XPermuted.map_partitions(lambda dataFrame: dataFrame.assign(**{feature: np.random.permutation(dataFrame[feature].to_numpy())}))     # Shuffle the feature values
        return self.scoreFunctionWithBatches(XPermuted)  # Evaluate score with the permuted feature

    def getFeatureImportanceDataframe(self, results: list) -> pd.DataFrame:
        """Create a Pandas DataFrame for importance values using CuPY"""
        importanceValues = cp.array(results)                # convert for GPU
        importancesMean = cp.mean(importanceValues, axis = 0)
        importancesStd = cp.std(importanceValues, axis = 0)

        importanceDataFrame = pd.DataFrame({
            'Feature': self.x.columns,
            'Importance': cp.asnumpy(importancesMean),      # transfer them back to the CPU
            'Std': cp.asnumpy(importancesStd)
        }).sort_values(by = 'Importance', ascending = False)

        logging.info("Feature importance DataFrame created.")
        logging.debug(f"Feature importance values: {importanceDataFrame}")

        return importanceDataFrame

    def plotFeatureImportance(self, featureImportanceDataFrame: pd.DataFrame) -> None:
        """Plot feature importance."""
        plt.figure(figsize=(10, 6))
        plt.barh(featureImportanceDataFrame['Feature'], featureImportanceDataFrame['Importance'], xerr = featureImportanceDataFrame['Std'], color = 'skyblue')

        plt.xlabel('Permutation Importance')
        plt.title('Feature Importance')

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f'featureImportancePlots/feature_importance_{timestamp}.png')
        plt.close()
        logging.info(f"Feature importance plot saved as feature_importance_{timestamp}.png")

    def evaluate(self, withDask: bool = True) -> None:
        """Evaluate feature importance and plot results."""
        try:
            featureImportanceDataFrame = self.computeFeatureImportanceDask() if withDask else self.computeFeatureImportance()
            self.plotFeatureImportance(featureImportanceDataFrame)

            logging.info("Feature importance evaluation completed.")
            logging.info("Feature Importance Data:\n" + featureImportanceDataFrame.to_string(index = False))

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            featureImportanceDataFrame.to_csv(f'featureImportancePlots/feature_importance_report_{timestamp}.csv', index = False)

        except Exception as e:
            logging.error(f"An error occurred while evaluating feature importance: {type(e).__name__} - {e}")


if __name__ == '__main__':
    modelPath = '../trainer/trainedModels/keras_model_test_dataset.keras'
    filePath = '../../data/higgs/prepared-higgs_test.csv'

        # Using Dask data frame
    dataLoaderDask = DataLoaderDask(filePath)
    dataFrame = dataLoaderDask.loadData()

    evaluator = FeatureImportanceEvaluator(modelPath, dataFrame)    # optionally set sampling rate and number of repetitions.
    evaluator.evaluate()        # evaluate() uses Dask by default
