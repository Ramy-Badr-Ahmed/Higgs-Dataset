import gzip
import time
import zipfile
import os
from tqdm import tqdm
import logging

logging.basicConfig(level = logging.INFO)

def extractZippedData(zipPath, extractTo):
    """
    Extract the contents of a zip dataset to the specified folder.
    Parameters:
    - zipPath (str): Path to the zip file.
    - extractTo (str): Folder path to extract the zip contents.
    """
    try:
        os.makedirs(extractTo, exist_ok = True)       # Ensure the extraction directory exists

        with zipfile.ZipFile(zipPath, 'r') as zipRef:
            fileNames = zipRef.namelist()        # Get list of files in the zip

            csvGzFileName = fileNames[0]
            csvGzPath = os.path.join(extractTo, csvGzFileName)

            if not os.path.isfile(csvGzPath):
                fileSize = zipRef.getinfo(csvGzFileName).file_size      # Get the size of the .gz file in the zip

                progressBar = tqdm(total = fileSize, unit = 'iB', unit_scale = True, unit_divisor = 1024, desc = "Extracting")

                with zipRef.open(csvGzFileName) as source, open(csvGzPath, 'wb') as target:
                    for chunk in iter(lambda: source.read(1024), b''):          # Read in chunks until the empty byte string
                        target.write(chunk)
                        progressBar.update(len(chunk))

                progressBar.close()
                logging.info(f"\tExtracted zip file to {csvGzPath}")
            else:
                logging.info(f"\tFile {csvGzPath} already exists. Skipping extraction.")

    except Exception as e:
        logging.error(f"\tAn error occurred while extracting the zip file: {e}")

def decompressGzFile(gzPath, outputPath):
    """
    Decompress a .gz file to a specified output path.
    Parameters:
    - gzPath (str): Path to the .gz file.
    - outputPath (str): Path to save the decompressed file.
    """
    startTime = time.time()
    logging.info(f"\tDecompressing ... {gzPath}")

    try:
        with gzip.open(gzPath, 'rb') as source:
            with open(outputPath, 'wb') as target:
                for chunk in iter(lambda: source.read(1024*1024), b''):  # Read in 1 MiB chunks until the empty byte string
                    target.write(chunk)

        elapsedTime = time.time() - startTime
        logging.info(f"\tDecompressed .gz file to {outputPath}")
        logging.info(f"\tTime taken: {elapsedTime:.2f} seconds")

    except Exception as e:
        logging.error(f"\tAn error occurred while decompressing the .gz file: {e}")

def cleanUp(gzCsvPath):
    """
    Deletes temporary files created during the process to free up space.
    Parameters:
    - gzCsvPath (str): Path to the extracted .gz file
    """
    try:
        if os.path.exists(gzCsvPath):
            os.remove(gzCsvPath)
            logging.info(f"\tTemporary .gz file cleaned up.")
    except Exception as e:
        logging.error(f"\tAn error occurred while cleaning up temporary file: {e}")

if __name__ == "__main__":

    zipDataUrl = 'https://archive.ics.uci.edu/static/public/280/higgs.zip'      # Higgs dataset URL

    # Paths for temporary and final files
    extractTo = '../higgs'

    zipPath = os.path.join(extractTo, 'higgs.zip')
    gzCsvPath = os.path.join(extractTo, 'higgs.csv.gz')
    finalCsvPath = os.path.join(extractTo, 'higgs.csv')

    extractZippedData(zipPath, extractTo)
    decompressGzFile(gzCsvPath, finalCsvPath)
    #cleanUp(gzCsvPath)
