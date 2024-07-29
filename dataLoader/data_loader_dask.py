import dask.dataframe as dd
import logging
from typing import Optional

logging.basicConfig(level = logging.INFO)

class DataLoaderDask:
    def __init__(self, filepath):
        self.filepath = filepath

    def loadData(self) -> Optional[dd.DataFrame]:
        """Load data from Higgs CSV file into a Dask DataFrame."""
        try:
            dataFrame = dd.read_csv(self.filepath, blocksize = '16MB')
            logging.info(f"\tData loaded successfully from {self.filepath}")
            logging.info(f"\tData shape: {dataFrame.shape}")

            rowsNumber = dataFrame.shape[0].compute()
            columnsNumber = dataFrame.shape[1]
            logging.info(f"\tData shape: ({rowsNumber}, {columnsNumber})")

            return dataFrame
        except Exception as e:
            logging.error(f"\tError loading data: {e}")
            return None

    def previewData(self, dataFrame: dd.DataFrame, n: int = 5) -> None:
        """Display the first n rows of the DataFrame."""
        if dataFrame is not None:
            logging.info("\tData preview:")
            logging.info(f"\n{dataFrame.head(n)}")
            logging.info("\nData types of each column:\n" + str(dataFrame.dtypes))
        else:
            logging.warning("\tNo data to preview.")

if __name__ == "__main__":

    filepath = '../data/higgs/prepared-higgs_test.csv'   # prepared-higgs_train.csv

    dataLoader = DataLoaderDask(filepath)

    dataFrame = dataLoader.loadData()

    dataLoader.previewData(dataFrame)
