import datetime
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import dask.dataframe as dd
import matplotlib.pyplot as plt
from tensorflow import keras
import logging

from dataLoader.data_loader_dask import DataLoaderDask

logging.basicConfig(level = logging.INFO)

class ModelEvaluator:
    def __init__(self, modelPath, dataFrame: dd.DataFrame):
        self.model = keras.models.load_model(modelPath)
        self.xTest = dataFrame.drop(columns=['target']).compute()
        self.yTest = dataFrame['target'].compute()
        os.makedirs('evaluationPlots', exist_ok = True)

    def evaluate(self) -> None:
        """Evaluate the model and display regression metrics."""
        try:
            predictions = self.model.predict(self.xTest)

            # Calculate metrics
            mse = mean_squared_error(self.yTest, predictions)
            mae = mean_absolute_error(self.yTest, predictions)
            r2 = r2_score(self.yTest, predictions)
            rmse = np.sqrt(mse)

            logging.info(f"Mean Squared Error: {mse:.4f}")
            logging.info(f"Mean Absolute Error: {mae:.4f}")
            logging.info(f"R-squared: {r2:.4f}")
            logging.info(f"Root Mean Squared Error: {rmse:.4f}")

            with open('evaluationPlots/metric_report.txt', 'w') as f:
                f.write(f"Mean Squared Error: {mse:.4f}\n"
                        f"Mean Absolute Error: {mae:.4f}\n"
                        f"R-squared: {r2:.4f}\n"
                        f"Root Mean Squared Error: {rmse:.4f}\n")

            # Plot actual vs. predicted values
            plt.figure(figsize=(10, 6))
            plt.scatter(self.yTest, predictions, alpha = 0.5)
            plt.plot([self.yTest.min(), self.yTest.max()],
                     [self.yTest.min(), self.yTest.max()], 'r--')  # Ideal line
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs. Predicted Values')
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(f'evaluationPlots/actual_vs_predicted__{timestamp}.png')
            plt.close()

            logging.info("\nModel evaluation completed.")

        except Exception as e:
            logging.error(f"\nAn error occurred while evaluating the model: {e}")


if __name__ == '__main__':
    modelPath = '../trainer/trainedModels/keras_model_trained_dataset.keras'
    filePath = '../../data/higgs/prepared-higgs_train.csv'

    # Using Dask data frame
    dataLoaderDask = DataLoaderDask(filePath)
    dataFrame = dataLoaderDask.loadData()

    evaluator = ModelEvaluator(modelPath, dataFrame)
    evaluator.evaluate()

