import os
import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
import seaborn as sns
import logging
from dataLoader.data_loader import DataLoader
from dataLoader.data_loader_dask import DataLoaderDask

logging.basicConfig(level = logging.INFO)

class EDA:
    def __init__(self, dataFrame: pd.DataFrame | dd.DataFrame):
        if not isinstance(dataFrame, (pd.DataFrame, dd.DataFrame)):
            raise ValueError("dataFrame must be a pandas or dask DataFrame.")

        self.dataFrame = dataFrame
        self.client = Client(n_workers = 10)
        os.makedirs('plots', exist_ok = True)

    def describeData(self):
        summaryStatistics = (
            self.dataFrame.describe(include = 'all')
            if isinstance(self.dataFrame, pd.DataFrame)
            else
            self.dataFrame.describe(include='all').compute()
        )
        logging.info("Summary statistics:\n" + str(summaryStatistics))

    def checkMissingValues(self):
        missingValues = (
            self.dataFrame.isnull().sum() if isinstance(self.dataFrame, pd.DataFrame)
            else
            self.dataFrame.isnull().sum().compute()
        )

        logging.info("Missing Values in Data:\n" + str(missingValues[missingValues > 0]))
        #logging.info("Detailed Missing Values in Data:\n" + str(missingValues))

    def visualiseFeatureCorrelation(self):
        """Creates a heatmap of the correlation matrix of the features."""
        plt.figure(figsize=(12, 10))

        correlationMatrix = self.dataFrame.corr()
        sns.heatmap(correlationMatrix, annot = True, fmt = ".2f", cmap = 'coolwarm',
                    square = True, cbar_kws={"shrink": .7}, annot_kws = {"size": 7})

        plt.title('Feature Correlation Matrix')
        plt.savefig('plots/correlation_heatmap.png')
        plt.show()

    def getCorrelationMatrix(self):
        """ returns correlation matrix of the dataset."""
        return self.dataFrame.corr()

    def visualizeTargetDistribution(self, targetName = 'target', sampleSizeFraction = 0.1):
        """Visualize the distribution of the target variable for class balance/imbalance. Samples data to 10% by default"""

        targetCounts = self.getCounts(targetName, sampleSizeFraction)

        plt.figure(figsize=(8, 6))
        sns.barplot(x = targetCounts.index, y = targetCounts.values)

        plt.title('Target Variable Distribution')
        plt.xlabel(targetName)
        plt.ylabel('Count')
        plt.xticks(rotation = 45)
        plt.tight_layout()

        plt.savefig('plots/target_distribution.png')
        plt.show()

    def getCounts(self, column, sampleSizeFraction):
        if not isinstance(self.dataFrame, dd.DataFrame):
            raise ValueError("Consider using Dask data frame only. Dataset is huge!")

        sampledDataFrame = self.dataFrame.sample(frac = sampleSizeFraction, random_state = 1)
        counts = sampledDataFrame[column].value_counts().compute()    # otherwise all data: self.dataFrame[column].value_counts().compute()
        return counts

    def visualizeFeatureDistribution(self, feature, sampleSizeFraction = 0.1):
        """Visualize the distribution of a specific feature."""
        if feature not in self.dataFrame.columns:
            logging.warning(f"Feature '{feature}' not found in the DataFrame.")
            return

        featureCounts = self.getCounts(feature, sampleSizeFraction)
        plt.figure(figsize = (8, 6))

        ax = sns.barplot(x = featureCounts.index.astype(str), y = featureCounts.values, color = 'skyblue', edgecolor = 'black')

        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title = feature)

        plt.tight_layout()
        plt.savefig(f'plots/{feature}_distribution.png')
        plt.show()

    def visualizeAllFeatureDistributions(self, columnsNumber = 10, maxFeatures = 20, sampleSizeFraction = 0.1):
        """Visualize distributions of all features (up to a maximum), with hue based on the target variable."""

        features = self.dataFrame.columns[:-1]  # Exclude target
        numberOfFeatures = min(len(features), maxFeatures)
        rowsNumber = (numberOfFeatures - 1) // columnsNumber + 1

        fig, axes = plt.subplots(rowsNumber, columnsNumber, figsize = (4 * columnsNumber, 3 * rowsNumber))
        axes = list(axes.flatten())

        sampledDataFrame = self.dataFrame.sample(frac = sampleSizeFraction, random_state = 1).compute()

        for i, feature in enumerate(features[:numberOfFeatures]):
            if feature not in sampledDataFrame.columns:
                continue

            sns.histplot(data = sampledDataFrame, x = feature, kde = True, ax = axes[i], stat = 'density', color = 'skyblue', edgecolor = 'black')

            axes[i].set_title(feature)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Density')
            axes[i].tick_params(axis = 'x', rotation = 45)
            axes[i].grid(True)

        for i in range(numberOfFeatures, len(axes)):
            axes[i].axis('off')         # Hide any remaining empty subplots

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title = 'Target', loc = 'center right', bbox_to_anchor = (1.05, 0.5))

        plt.tight_layout()
        plt.savefig('plots/all_feature_distributions.png')
        plt.show()

    def visualizeFeatureScatter(self, feature1, feature2):
        """Visualize scatter plot between two features."""
        if feature1 not in self.dataFrame.columns or feature2 not in self.dataFrame.columns:
            logging.warning(f"One or both features '{feature1}' and '{feature2}' not found in the DataFrame.")
            return

        sampledDataFrame = self.dataFrame.sample(frac=0.01, random_state=1).compute()

        plt.figure(figsize=(8, 6))
        plt.scatter(sampledDataFrame[feature1], sampledDataFrame[feature2], alpha=0.5)
        plt.title(f'Scatter plot between {feature1} and {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.savefig(f'plots/scatter_{feature1}_{feature2}.png')
        plt.show()

    def visualizePairplot(self, sampleSizeFraction = 0.1):
        """Visualize pairplot for the first few features in the dataset."""
        if not isinstance(self.dataFrame, dd.DataFrame):
            raise ValueError("Consider using Dask data frame only.")

        sampledDataFrame = self.dataFrame.sample(frac=sampleSizeFraction, random_state=1).compute()
        sns.pairplot(sampledDataFrame)
        plt.title("Pairplot of Features")
        plt.savefig('plots/pairplot.png')
        plt.show()

    def visualizeFeatureBoxplot(self, feature):
        """Visualize the boxplot of a specific feature."""
        if feature not in self.dataFrame.columns:
            logging.warning(f"Feature '{feature}' not found in the DataFrame.")
            return

        plt.figure(figsize=(8, 6))
        sns.boxplot(x = self.dataFrame[feature].compute(), color = 'skyblue')
        plt.title(f'Boxplot of {feature}')
        plt.xlabel(feature)
        plt.savefig(f'plots/{feature}_boxplot.png')
        plt.show()


if __name__ == '__main__':
    filepath = '../data/higgs/prepared-higgs_test.csv'   # prepared-higgs_train.csv

        # using Pandas data frame
    #dataLoader = DataLoader(filepath)
    #dataFrame = dataLoader.loadData()

        # using Dask data frame
    dataLoaderDask = DataLoaderDask(filepath)
    dataFrame = dataLoaderDask.loadData()

    if dataFrame is not None:
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