import requests
import os
from tqdm import tqdm
import logging

logging.basicConfig(level = logging.INFO)

def downloadDataset(dataSetURL, dataSetPath):
    """
    Download a dataset file from the specified URL and save it locally with a progress bar.
    Parameters:
    - dataSetURL (str): URL of the file.
    - dataSetPath (str): Path to save the file locally.
    """
    try:
        if not os.path.exists(dataSetPath):
            response = requests.get(dataSetURL, stream = True)
            totalSize = int(response.headers.get('content-length', 0))
            blockSize = 1024  # 1 Kibibyte

            progressBar = tqdm(total = totalSize, unit = 'iB', unit_scale = True, unit_divisor = 1024, desc = "Downloading")

            with open(dataSetPath, 'wb') as file:
                for chunk in response.iter_content(blockSize):
                    if chunk:
                        progressBar.update(len(chunk))
                        file.write(chunk)

            progressBar.close()
            logging.info(f"\tDownloaded file to {dataSetPath}")
        else:
            logging.info(f"\tFile already exists at {dataSetPath}")
    except Exception as e:
        logging.error(f"\tAn error occurred while downloading the file: {e}")

def cleanUp(zipPath):
    """
    Deletes temporary files created during the process to free up space.
    Parameters:
    - zipPath (str): Path to the downloaded zip file.
    """
    try:
        if os.path.exists(zipPath):
            os.remove(zipPath)
            logging.info(f"\tTemporary zip file cleaned up.")
    except Exception as e:
        logging.error(f"\tAn error occurred while cleaning up temporary file: {e}")

if __name__ == "__main__":

    zipDataUrl = 'https://archive.ics.uci.edu/static/public/280/higgs.zip'      # Higgs dataset URL

    zipPath = '../higgs/higgs.zip'

    downloadDataset(zipDataUrl, zipPath)
    #cleanUp(zipPath)



