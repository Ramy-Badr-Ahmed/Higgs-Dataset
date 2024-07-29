import os
import logging
import dask.dataframe as dd
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from dataLoader.data_loader_dask import DataLoaderDask
import matplotlib.pyplot as plt
import datetime

logging.basicConfig(level = logging.INFO)

class ModelTrainer:
    def __init__(self, dataFrame: dd.DataFrame, params: dict = None):
        self.dataFrame = dataFrame
        self.model = None
        self.history = None

        defaultParams = {
            "epochs": 10,
            "batchSize": 32,
            "minSampleSize": 100000,
            "learningRate": 0.001,
            "modelBuilder": None,    # callable. Fallback model in defineModel()
            "loss": 'mean_squared_error',
            "metrics": ['mae']
        }

        self.params = {**defaultParams, **(params or {})}       # use default parameters; otherwise merge with provided params
        self.validateParameters(self.params)

        # unpack model parameters
        self.epochs = self.params["epochs"]
        self.batchSize = self.params["batchSize"]
        self.minSampleSize = self.params["minSampleSize"]
        self.learningRate = self.params["learningRate"]
        self.modelBuilder = self.params["modelBuilder"]
        self.loss = self.params["loss"]
        self.metrics = self.params["metrics"]

        os.makedirs('trainedModels', exist_ok = True)

    def validateParameters(self, params: dict) -> None:
        epochs = params.get("epochs", 10)
        batchSize = params.get("batchSize", 32)
        minSampleSize = params.get("minSampleSize", 100000)
        learningRate = params.get("learningRate", 0.001)

        if epochs <= 0:
            raise ValueError("Epochs must be a positive integer.")
        if batchSize <= 0:
            raise ValueError("Batch size must be a positive integer.")
        if minSampleSize <= 0:
            raise ValueError("Minimum sample size must be a positive integer.")
        if learningRate <= 0:
            raise ValueError("Learning rate must be a positive float.")

    def trainKerasModel(self, sample: bool = False, frac: float = 0.1) -> None:
        """Train a Keras model on the HIGGS dataset."""
        try:
            xTrain, xTest, yTrain, yTest = self.prepareData(sample, frac)
            if xTrain is None or xTest is None:
                return

            self.defineModel(xTrain.shape[1])
            self.trainModel(xTrain, yTrain, xTest, yTest)
            self.evaluateModel(xTest, yTest)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.saveModel(f'trainedModels/keras_model_{timestamp}.keras')

        except Exception as e:
            logging.error(f"An error occurred while training the Keras model: {e}")

    def prepareData(self, sample: bool = False, frac: float = 0.1) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
        """Prepare the training and testing data."""
        if sample:
            X, y = self.sampleData(frac=frac)
            if X is None or y is None:
                logging.error("No valid data to train on. Exiting.")
                return None, None
        else:
            X = self.dataFrame.drop('target', axis=1)
            y = self.dataFrame['target']

        # Convert Dask DataFrame to Pandas DataFrame for Keras
        X = X.compute()
        y = y.compute()

        if len(X) < self.minSampleSize:
            logging.error("Sampled data is too small for training.")
            return None, None

        # Split the dataset
        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 42)
        return xTrain, xTest, yTrain, yTest

    def sampleData(self, frac: float = 0.1) -> (pd.DataFrame, pd.Series):
        """Sample a fraction of the data and return X and y as Dask DataFrames."""
        sampledData = self.dataFrame.sample(frac = frac, random_state = 42)

        sampled_size = sampledData.shape[0].compute()
        if sampled_size == 0:
            logging.error("Sampled data is empty.")
            return None, None

        xSample = sampledData.drop('target', axis = 1)
        ySample = sampledData['target']
        return xSample, ySample

    def defineModel(self, inputShape: int) -> None:
        if self.modelBuilder:
            self.model = self.modelBuilder(inputShape)
        else:
            self.model = keras.Sequential([          # Fallback to this default model for regression
                layers.Input(shape=(inputShape,)),
                layers.Dense(128, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(1)
            ])

        self.printModelSummary()

        optimizer = keras.optimizers.Adam(learning_rate = self.learningRate)
        self.model.compile(optimizer = optimizer, loss = self.loss, metrics = self.metrics)       # default: 'mean_squared_error', ['mae']

    def printModelSummary(self):
        if self.model:
            self.model.summary()
            logging.info(f"Total number of layers: {len(self.model.layers)}")
        else:
            logging.warning("Model is not defined.")

    def trainModel(self, xTrain: pd.DataFrame, yTrain: pd.Series, xTest: pd.DataFrame, yTest: pd.Series) -> None:
        logging.info(f"xTrain shape: {xTrain.shape}, yTrain shape: {yTrain.shape}")

        self.history = self.model.fit(xTrain, yTrain, epochs = self.epochs,
                                      batch_size = self.batchSize, validation_data = (xTest, yTest))

        for epoch in range(self.epochs):
            logging.info(f"Epoch {epoch + 1}/{self.epochs} -->"
                         f" (Loss: {self.history.history['loss'][epoch]}, Val Loss: {self.history.history['val_loss'][epoch]})")


    def evaluateModel(self, xTest: pd.DataFrame, yTest: pd.Series) -> None:
        loss, mae = self.model.evaluate(xTest, yTest)
        logging.info(f"\n\tKeras Model Mean Absolute Error: {mae}")
        logging.info(f"\n\tKeras Model loss: {loss}")

    def plotTrainingHistory(self) -> None:
        """Plot training and validation accuracy and loss."""
        if self.history is not None:
            plt.figure(figsize=(12, 6))

            # Plot the training and validation loss
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')

            # Plot 'Mean Absolute Error' if it is part of the history
            if 'mae' in self.history.history:
                plt.plot(self.history.history['mae'], label='Training MAE')
                plt.plot(self.history.history['val_mae'], label='Validation MAE')

            plt.title('Loss over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss / MAE')
            plt.legend()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(f'trainedModels/training_loss_{timestamp}.png')
            plt.show()
        else:
            logging.warning("No training history to plot.")

    def saveModel(self, modelPath) -> None:
        if isinstance(self.model, keras.Model):
            self.model.save(modelPath)
        else:
            joblib.dump(self.model, modelPath)
        logging.info(f"Model trained and saved to {modelPath}")

def customModel(inputShape: int) -> Model:
    """Example of a custom model builder function for regression"""
    model = keras.Sequential([
        layers.Input(shape=(inputShape,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    return model

if __name__ == '__main__':
    filePath = '../../data/higgs/prepared-higgs_test.csv'

        # Using Dask data frame
    dataLoaderDask = DataLoaderDask(filePath)
    dataFrame = dataLoaderDask.loadData()

    if dataFrame is not None:
        # Optional: Define your model training/compiling/defining parameters as a dictionary and pass it to the class constructor
        params = {
            "epochs": 10,
            "batchSize": 32,
            "minSampleSize": 100000,
            "learningRate": 0.001,
            "modelBuilder": customModel,     # callable
            "loss": 'mean_absolute_error',
            "metrics": ['mae']
        }
        trainer = ModelTrainer(dataFrame)
        trainer.trainKerasModel()           # optional: Train the Keras model with sampling, Set: trainKerasModel(sample = true, frac = 0.1).
        trainer.plotTrainingHistory()
