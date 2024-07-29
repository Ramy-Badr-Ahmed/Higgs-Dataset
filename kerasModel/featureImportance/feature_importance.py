import os
import logging
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from dataLoader.data_loader_dask import DataLoaderDask

logging.basicConfig(level=logging.INFO)

class FeatureImportanceEvaluator:
    def __init__(self, modelPath: str, dataFrame: dd.DataFrame):
        self.model = keras.models.load_model(modelPath)
        self.X = dataFrame.drop(columns=['target']).compute()
        self.y = dataFrame['target'].compute()
        os.makedirs('featureImportancePlots', exist_ok = True)

    def computeFeatureImportance(self) -> pd.DataFrame:
        """Compute feature importance using permutation importance."""

        def score_function(model, X, y):
            predictions = model.predict(X)
            return mean_squared_error(y, predictions)

        baselineScore = score_function(self.model, self.X, self.y)
        logging.info(f"Baseline Score (MSE): {baselineScore}")

        # # Sample 10% of your data
        # sampled_X = self.X.sample(frac=0.01, random_state=42)
        # sampled_y = self.y.sample(frac=0.01, random_state=42)
        # result = permutation_importance(self.model, sampled_X, sampled_y, n_repeats=30, random_state=42, scoring = score_function)

        result = permutation_importance(self.model, self.X, self.y, n_repeats = 30, random_state = 42, scoring = score_function)

        # Create a DataFrame for importance values
        featureImportanceDataFrame = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': result.importances_mean,
            'Std': result.importances_std
        }).sort_values(by = 'Importance', ascending = False)

        return featureImportanceDataFrame

    def plotFeatureImportance(self, featureImportanceDataFrame) -> None:
        """Plot feature importance."""
        plt.figure(figsize=(10, 6))
        plt.barh(featureImportanceDataFrame['Feature'], featureImportanceDataFrame['Importance'],
                 xerr=featureImportanceDataFrame['Std'], color = 'skyblue')
        plt.xlabel('Permutation Importance')
        plt.title('Feature Importance')
        plt.savefig('featureImportancePlots/feature_importance.png')
        plt.close()

    def evaluate(self) -> None:
        """Evaluate feature importance and plot results."""
        try:
            featureImportanceDataFrame = self.computeFeatureImportance()
            self.plotFeatureImportance(featureImportanceDataFrame)

            logging.info("Feature importance evaluation completed.")
            logging.info("Feature Importance Data:\n" + featureImportanceDataFrame.to_string(index = False))

            featureImportanceDataFrame.to_csv('featureImportancePlots/feature_importance_report.csv', index = False)

        except Exception as e:
            logging.error(f"An error occurred while evaluating feature importance: {e}")


if __name__ == '__main__':
    modelPath = '../trainer/trainedModels/keras_model_test_dataset.keras'
    filePath = '../../data/higgs/prepared-higgs_test.csv'

    # Using Dask data frame
    dataLoaderDask = DataLoaderDask(filePath)
    dataFrame = dataLoaderDask.loadData()

    evaluator = FeatureImportanceEvaluator(modelPath, dataFrame)
    evaluator.evaluate()
