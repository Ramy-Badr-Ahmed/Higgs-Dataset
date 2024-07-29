import os
from download_data import downloadDataset, cleanUp
from data_extraction import extractZippedData, decompressGzFile, cleanUp
from data_preparation import prepareData, cleanUp

def main():
    zipDataUrl = 'https://archive.ics.uci.edu/static/public/280/higgs.zip'
    extractTo = '../higgs'
    zipPath = os.path.join(extractTo, 'higgs.zip')
    gzCsvPath = os.path.join(extractTo, 'higgs.csv.gz')
    finalCsvPath = os.path.join(extractTo, 'higgs.csv')
    preparedCsvPath = os.path.join(extractTo, 'prepared-higgs.csv')

    downloadDataset(zipDataUrl, zipPath)

    extractZippedData(zipPath, extractTo)
    decompressGzFile(gzCsvPath, finalCsvPath)

    prepareData(finalCsvPath, preparedCsvPath)

    cleanUp(zipPath)
    cleanUp(gzCsvPath)
    cleanUp(finalCsvPath)

if __name__ == "__main__":
    main()