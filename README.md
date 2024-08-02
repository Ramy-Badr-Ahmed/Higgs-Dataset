![Python](https://img.shields.io/badge/Python-3670A0?style=plastic&logo=python&logoColor=ffdd54)  

![GitHub](https://img.shields.io/github/license/Ramy-Badr-Ahmed/Higgs-Dataset-Training)

# Model Training and Evaluation for Higgs Dataset

## Overview

This repository demonstrates training and evaluating a Keras model using the Higgs dataset available from the UCI ML Repository.

> [Higgs Dataset](http://archive.ics.uci.edu/ml/datasets/HIGGS)

The dataset has been studied in this publication:

> [Searching for Exotic Particles in High-energy Physics with Deep Learning.<br>Baldi, P., P. Sadowski, and D. Whiteson. 
Nature Communications 5, 4308 (2014)](https://www.nature.com/articles/ncomms5308)

The ML pipeline includes downloading the dataset, data preparation, model training, evaluation, feature importance analysis, and visualization of results. Dask is utilised for handling this large datasets for parallel processing.

### Installation

1) Create and source virtual environment:
```shell
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```
2) Install the dependencies:
```shell
pip install -r requirements.txt
```

### Data

The Higgs dataset can be downloaded directly from the provided scripts in separate steps

- `download_data.py`  ~ 2.6 GB
- `data_extraction.py` ~ 7 GB
- `data_preparation.py` ~ test dataset: 240 MB, trained dataset: 5 GB

Alternatively, you can run directly the main script from the `data/src/main.py`:

```shell
python data/src/main.py
```

#### Downloading Data
Download a dataset file from the specified URL with a progress bar.

##### Script
```shell
python data/download_data.py
```

##### Example Usage
```python
zipDataUrl = 'https://archive.ics.uci.edu/static/public/280/higgs.zip'      # Higgs dataset URL
zipPath = '../higgs/higgs.zip'
downloadDataset(zipDataUrl, zipPath)
cleanUp(zipPath)        # Clean up downloaded zip file (~ 2.6 GB)
```

#### Data Extraction
Extract the contents of a zip dataset and decompress the .gz dataset file to a specified output path.

##### Script
```shell
python data/data_extraction.py
```

##### Example Usage
```python
zipDataUrl = 'https://archive.ics.uci.edu/static/public/280/higgs.zip'      # Higgs dataset URL
extractTo = '../higgs'
zipPath = os.path.join(extractTo, 'higgs.zip')
gzCsvPath = os.path.join(extractTo, 'higgs.csv.gz')
finalCsvPath = os.path.join(extractTo, 'higgs.csv')

extractZippedData(zipPath, extractTo)
decompressGzFile(gzCsvPath, finalCsvPath)
cleanUp(gzCsvPath)      # Clean up gzipped file (~ 2.6 GB)
```

#### Data Preparation
Set column names and separates the test set from the training data based on the dataset description (500,000 test sets).

##### Script
```shell
python data/data_preparation.py
```

##### Example Usage
```python
prepareFrom = '../higgs'
csvPath = os.path.join(prepareFrom, 'higgs.csv')
preparedCsvPath = os.path.join(prepareFrom, 'prepared-higgs.csv')
prepareData(csvPath, preparedCsvPath)
cleanUp(csvPath)         # Clean up gzipped file (~ 7.5 GB)
```

### Loading Data

#### Using Pandas

Use the `dataLoader/data_loader.py` script to load the prepared dataset into a Pandas DataFrame.

##### Script
```shell
python data/src/data_loader.py
```

##### Example Usage
```python
filepath = '../data/higgs/prepared-higgs_train.csv'   # prepared-higgs_test.csv
dataLoader = DataLoader(filepath)
dataFrame = dataLoader.loadData()
dataLoader.previewData(dataFrame)
```

#### Using Dask

Use the `dataLoader/data_loader_dask.py` script to load the prepared dataset into a Dask DataFrame, which is beneficial for this large dataset.

##### Script
```shell
python data/src/data_loader_dask.py
```
##### Example Usage:
```python
filepath = '../data/higgs/prepared-higgs_train.csv'   # prepared-higgs_test.csv
dataLoader = DataLoaderDask(filepath)
dataFrame = dataLoader.loadData()
dataLoader.previewData(dataFrame)
```

### Exploratory Data Analysis (EDA)

Provides various functions for performing EDA, including visualising correlations, checking missing values, and plotting feature distributions.
The data analysis plots are saved under `eda/plots`.

##### Script
```shell
python exploration/eda.py
```

##### Example Usage:
```python
filepath = '../data/higgs/prepared-higgs_train.csv'   # prepared-higgs_test.csv

    # using Dask data frame
dataLoaderDask = DataLoaderDask(filepath)
dataFrame = dataLoaderDask.loadData()

eda = EDA(dataFrame)
eda.describeData()
eda.checkMissingValues()
eda.visualiseFeatureCorrelation()

eda.visualizeTargetDistribution()
eda.visualizeFeatureDistribution('feature_1')
eda.visualizeAllFeatureDistributions()
eda.visualizeFeatureScatter('feature_1', 'feature_2')
eda.visualizeTargetDistribution()
eda.visualizeFeatureBoxplot('feature_2')
```

### Usage

#### Training the Model
The model is defined using Keras with the following default architecture for binary classification:

- Input layer with 128 neurons (dense)
- Hidden layer with 64 neurons (dense)
- Output layer with 1 neuron (activation function: sigmoid)

You can customize the model architecture by providing a different modelBuilder function in the ModelTrainer class.

The trained models and training loss plots are saved under `kerasModel/trainer/trainedModels`.

##### Script
```shell
python kerasModel/trainer/model_trainer.py
```

##### Example Usage:
```python
filePath = '../../data/higgs/prepared-higgs_train.csv'

def customModel(inputShape: int) -> Model:
    """Example of a custom model builder function for classification"""
    model = keras.Sequential([
        layers.Input(shape=(inputShape,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    return model
        
dataLoaderDask = DataLoaderDask(filePath)
dataFrame = dataLoaderDask.loadData()

## Optional: Define model training/compiling/defining parameters as a dictionary and pass it to the class constructor
params = {
    "epochs": 10,
    "batchSize": 32,
    "minSampleSize": 100000,
    "learningRate": 0.001,
    "modelBuilder": customModel,     # callable
    "loss": 'binary_crossentropy',    
    "metrics": ['accuracy']
}
trainer = ModelTrainer(dataFrame, params)
trainer.trainKerasModel()           # optional: Train the Keras model with sampling, Set: trainKerasModel(sample = true, frac = 0.1).
trainer.plotTrainingHistory()
```

#### Evaluating the Model
The evaluation script computes metrics like:

- Accuracy
- Precision
- Recall (Sensitivity
- F1 Score
- Classification Report

The evaluation includes visualizations such as
- Confusion Matrix
- ROC Curve

The evaluation results are logged and saved to a file under `kerasModel/evaluator/evaluationPlots`.

##### Script
```shell
python kerasModel/evaluator/model_evaluator.py
```

##### Example Usage:
```python
modelPath = '../trainer/trainedModels/keras_model_trained_dataset.keras'
filePath = '../../data/higgs/prepared-higgs_train.csv'

dataLoaderDask = DataLoaderDask(filePath)
dataFrame = dataLoaderDask.loadData()

evaluator = ModelEvaluator(modelPath, dataFrame)
evaluator.evaluate()
```

#### Feature Importance Analysis

The feature importance is computed using permutation importance and visualised using a bar chart. It is implemented once using the Pandas approach (with SciKit) and another using Dask for parallel processing.

The chart and the result CSV file are saved under `kerasModel/featureImportance/featureImportancePlots`.

##### Script
```shell
python kerasModel/featureImportance/feature_importance.py
```

##### Example Usage:
```python
modelPath = '../trainer/trainedModels/keras_model_test_dataset.keras'
filePath = '../../data/higgs/prepared-higgs_test.csv'

dataLoaderDask = DataLoaderDask(filePath)
dataFrame = dataLoaderDask.loadData()

evaluator = FeatureImportanceEvaluator(modelPath, dataFrame)
evaluator.evaluate()
        
        # Alternatively
evaluator = FeatureImportanceEvaluator(modelPath, dataFrame, sampleFraction = 0.1, nRepeats=32)  # with sampling
evaluator.evaluate(withDask = False)        # with pandas
```
