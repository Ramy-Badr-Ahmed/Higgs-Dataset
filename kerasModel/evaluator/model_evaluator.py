import datetime
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_auc_score, roc_curve
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

            predictedClasses = (predictions > 0.5).astype(int)

            # Calculate metrics
            accuracy = accuracy_score(self.yTest, predictedClasses)
            precision = precision_score(self.yTest, predictedClasses)
            recall = recall_score(self.yTest, predictedClasses)
            f1 = f1_score(self.yTest, predictedClasses)
            report = classification_report(self.yTest, predictedClasses)
            confusionMatrix = confusion_matrix(self.yTest, predictedClasses)
            auc = roc_auc_score(self.yTest, predictions)
            fpr, tpr, thresholds = roc_curve(self.yTest, predictions)

            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"F1 Score: {f1:.4f}")
            logging.info(f"AUC: {auc}")
            logging.info(f"Classification Report:\n{report}")
            logging.info(f"Confusion Matrix:\n{confusionMatrix}")

            self.saveMetrics(accuracy, precision, recall, f1, report, confusionMatrix, auc)

            self.plotEvaluations(confusionMatrix)
            self.plotROC(fpr, tpr, auc)

            logging.info("\nModel evaluation completed.")

        except Exception as e:
            logging.error(f"\nAn error occurred while evaluating the model: {e}")

    def saveMetrics(self, accuracy, precision, recall, f1, auc, report, confusionMatrix) -> None:
        with open('evaluationPlots/metric_report.txt', 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"F1 Score: {f1:.4f}\n"
            f"AUC: {auc}\n"
            f"Classification Report:{report}\n"
            f"Confusion Matrix:{confusionMatrix}")

    def plotEvaluations(self, confusionMatrix) -> None:
        plt.figure(figsize = (8, 6))
        plt.imshow(confusionMatrix, interpolation = 'nearest', cmap = plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Background', 'Signal'])
        plt.yticks(tick_marks, ['Background', 'Signal'])

        threshold = confusionMatrix.max() / 2.
        for i, j in np.ndindex(confusionMatrix.shape):
            plt.text(j, i, format(confusionMatrix[i, j], 'd'),
                     horizontalalignment = "center",
                     color = "white" if confusionMatrix[i, j] > threshold else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f'evaluationPlots/confusion_matrix_{timestamp}.png')
        plt.close()

    def plotROC(self, fpr, tpr, auc) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc = 'lower right')
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f'evaluationPlots/roc_curve_{timestamp}.png')
        plt.close()


if __name__ == '__main__':
    modelPath = '../trainer/trainedModels/keras_model_trained_dataset.keras'
    filePath = '../../data/higgs/prepared-higgs_train.csv'

    # Using Dask data frame
    dataLoaderDask = DataLoaderDask(filePath)
    dataFrame = dataLoaderDask.loadData()

    evaluator = ModelEvaluator(modelPath, dataFrame)
    evaluator.evaluate()

