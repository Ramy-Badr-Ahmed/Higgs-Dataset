import os
import pandas as pd
import time
import logging

logging.basicConfig(level = logging.INFO)

def prepareData(csvPath, outputPath, testSetSize=500000):
    """
    Load the dataset from a CSV file, set column names, and save it to a local file.
    Also separates the test set from the training data based on the dataset description (500,000 test sets)

    Parameters:
    - csvPath (str): Path to the CSV file.
    - outputPath (str): Path to save the dataset locally.
    - testSetSize (int): Number of examples to be used as the test set.
    """
    startTime = time.time()
    logging.info(f"\tPreparing data ... ")

    try:
        datasetDataFrame = pd.read_csv(csvPath, header = None)

        logging.info(f"\tSetting column names based on the dataset description")
        columns = [f'feature_{i+1}' for i in range(21)] + [f'derived_feature_{i+1}' for i in range(7)] + ['target']
        datasetDataFrame.columns = columns

        logging.info(f"\tSeparating the test set from the main dataset")
        if testSetSize > 0:
            # Ensure the dataset is large enough
            if len(datasetDataFrame) > testSetSize:
                trainDataFrame = datasetDataFrame[:-testSetSize]
                testDataFrame = datasetDataFrame[-testSetSize:]

                # Save the datasets
                trainDataFrame.to_csv(outputPath.replace('.csv', '_train.csv'), index = False)
                testDataFrame.to_csv(outputPath.replace('.csv', '_test.csv'), index = False)

                logging.info(f"\tDataset successfully split into training and test sets.")
            else:
                # If not enough data, save it all as training
                datasetDataFrame.to_csv(outputPath, index=False)
                logging.info(f"\tDataset saved to {outputPath}, but no split was performed due to insufficient data.")
        else:
            # No test set separation
            datasetDataFrame.to_csv(outputPath, index = False)
            logging.info(f"\tDataset successfully saved to {outputPath}")

        elapsedTime = time.time() - startTime
        logging.info(f"\tTime taken for loading and saving: {elapsedTime:.2f} seconds")

    except Exception as e:
        logging.info(f"\tAn error occurred while preparing the data: {e}")

def cleanUp(csvPath):
    """
    Deletes temporary files created during the process to free up space.
    Parameters:
    - csvPath (str): Path to temporary csv file of the unprepared train and test data
    """
    try:
        if os.path.exists(csvPath):
            os.remove(csvPath)
            logging.info(f"\tUnprepared .csv file cleaned up.")
    except Exception as e:
        logging.error(f"\tAn error occurred while cleaning up temporary file: {e}")

if __name__ == "__main__":

    prepareFrom = '../higgs'

    csvPath = os.path.join(prepareFrom, 'higgs.csv')
    preparedCsvPath = os.path.join(prepareFrom, 'prepared-higgs.csv')

    prepareData(csvPath, preparedCsvPath)
    #cleanUp(csvPath)