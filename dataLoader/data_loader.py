import pandas as pd
import logging
from typing import Optional

logging.basicConfig(level = logging.INFO)

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def loadData(self)-> Optional[pd.DataFrame]:
        """Load data from Higgs CSV file into a Pandas DataFrame."""
        try:
            dataFrame = pd.read_csv(self.filepath)
            logging.info(f"\tData loaded successfully from {self.filepath}")
            logging.info(f"\tData shape: {dataFrame.shape}")
            return dataFrame
        except Exception as e:
            logging.error(f"\tError loading data: {e}")
            return None

    def previewData(self, dataFrame: pd.DataFrame,  n: int = 5) -> None:
        """Display the first n rows of the DataFrame."""
        if dataFrame is not None:
            logging.info("\tData preview:")
            logging.info(f"\n{dataFrame.head(n)}")
            logging.info("\nData types of each column:\n" + str(dataFrame.dtypes))
        else:
            logging.warning("\tNo data to preview.")

if __name__ == "__main__":

    filepath = '../data/higgs/prepared-higgs_train.csv'   # prepared-higgs_test.csv

    dataLoader = DataLoader(filepath)

    dataFrame = dataLoader.loadData()

    dataLoader.previewData(dataFrame)