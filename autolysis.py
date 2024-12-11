# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "scikit-learn",
#   "numpy",
#   "chardet",
#   "requests",
#   "tqdm"
# ]
# ///
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np
import chardet
import requests
import logging
from tqdm import tqdm

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration for AI Proxy API
SETTINGS = {
    "PROXY_ENDPOINT": "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
    "RESULTS_DIR": os.path.dirname(os.path.abspath(__file__)),
    "MAX_IMAGES": 3,
    "CHUNK_SIZE": 10000
}

class DataAnalysisTool:
    def __init__(self):
        self.image_counter = 0

    def get_proxy_key(self):
        try:
            return os.environ["AIPROXY_TOKEN"]
        except KeyError:
            logging.error("AIPROXY_TOKEN environment variable not set.")
            sys.exit(1)

    def query_ai(self, prompt, context):
        proxy_key = self.get_proxy_key()
        try:
            api_headers = {"Authorization": f"Bearer {proxy_key}", "Content-Type": "application/json"}
            request_body = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": f"{prompt}\nContext:\n{context}"}]
            }
            response = requests.post(SETTINGS["PROXY_ENDPOINT"], headers=api_headers, json=request_body)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as error:
            logging.error(f"AI Proxy communication error: {error}")
            sys.exit(1)

    def save_plot(self, filename):
        if self.image_counter >= SETTINGS["MAX_IMAGES"]:
            logging.warning("Image generation limit reached.")
            return
        try:
            plt.gcf().set_size_inches(5.12, 5.12)
            plt.tight_layout()
            plt.savefig(
                os.path.join(SETTINGS["RESULTS_DIR"], filename),
                bbox_inches='tight',
                dpi=100
            )
            plt.close()
            self.image_counter += 1
        except Exception as error:
            logging.error(f"Error saving plot {filename}: {error}")

    def get_file_encoding(self, filepath):
        try:
            with open(filepath, 'rb') as file:
                raw_data = file.read()
            detected_encoding = chardet.detect(raw_data)
            return detected_encoding['encoding']
        except Exception as error:
            logging.error(f"Error detecting encoding for file {filepath}: {error}")
            sys.exit(1)

    def review_missing_values(self, dataframe):
        missing_counts = dataframe.isnull().sum()
        missing_percentages = (missing_counts / len(dataframe)) * 100
        missing_summary = missing_percentages[missing_percentages > 0].sort_values(ascending=False)
        if not missing_summary.empty:
            plt.figure()
            sns.barplot(x=missing_summary.index, y=missing_summary.values, color="lightblue")
            plt.xticks(rotation=45, ha='right')
            plt.title("Columns with Missing Data (%)")
            plt.xlabel("Columns")
            plt.ylabel("Missing Percentage")
            self.save_plot("missing_data_report.png")
        return missing_summary

    def evaluate_correlation(self, dataframe):
        numeric_data = dataframe.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            correlations = numeric_data.corr()
            plt.figure()
            sns.heatmap(correlations, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Analysis")
            self.save_plot("correlation_matrix.png")
            return correlations
        return None

    def identify_outliers(self, dataframe):
        numeric_data = dataframe.select_dtypes(include=[np.number])
        outlier_data = {}
        for column in numeric_data.columns:
            q1 = numeric_data[column].quantile(0.25)
            q3 = numeric_data[column].quantile(0.75)
            iqr = q3 - q1
            outlier_data[column] = numeric_data[(numeric_data[column] < q1 - 1.5 * iqr) | (numeric_data[column] > q3 + 1.5 * iqr)].shape[0]
        return outlier_data

    def perform_data_clustering(self, dataframe):
        numeric_data = dataframe.select_dtypes(include=[np.number])
        if not numeric_data.empty and numeric_data.shape[1] > 1:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(numeric_data.fillna(0))
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(reduced_data)
            silhouette_avg = silhouette_score(reduced_data, clusters)
            dataframe['Cluster_Group'] = clusters

            plt.figure()
            sns.scatterplot(
                x=reduced_data[:, 0], y=reduced_data[:, 1],
                hue=clusters, palette="viridis", s=50
            )
            plt.title(f"Cluster Representation (Silhouette Score: {silhouette_avg:.2f})")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            self.save_plot("clusters.png")

            return dataframe['Cluster_Group'].value_counts()
        return None

    def explore_distributions(self, dataframe):
        numeric_columns = dataframe.select_dtypes(include=[np.number])
        for column in numeric_columns.columns:
            if self.image_counter >= SETTINGS["MAX_IMAGES"]:
                break
            plt.figure()
            sns.histplot(dataframe[column], kde=True, bins=30, color="darkblue")
            plt.title(f"Data Distribution: {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            self.save_plot(f"{column}_distribution.png")

    def feature_engineering(self, dataframe):
        logging.info("Performing feature engineering...")
        # Example: Adding interaction terms or binning variables
        dataframe['total_columns'] = dataframe.shape[1]
        # Add your specific feature engineering logic here
        return dataframe

    def create_readme(self, dataframe, missing_data, correlations, outliers, clusters):
        analysis_context = f"""
        Dataset Summary:
        Total records: {dataframe.shape[0]}
        Total fields: {dataframe.shape[1]}
        Column names: {', '.join(dataframe.columns)}

        Missing Values:
        {missing_data}

        Correlation Details:
        {correlations}

        Outliers Found:
        {outliers}

        Cluster Analysis:
        {clusters}
        """
        summary_story = self.query_ai(
            "Craft a detailed and structured narrative based on dataset analysis insights.",
            analysis_context
        )

        further_analysis = self.query_ai(
            "Propose additional analysis steps based on the current insights.", analysis_context
        )

        try:
            readme_path = os.path.join(SETTINGS["RESULTS_DIR"], "README.md")
            with open(readme_path, "w") as file:
                file.write("# Analysis Report\n\n")
                file.write(f"## Dataset Overview\n{analysis_context}\n\n")
                file.write("## Additional Insights\n")
                file.write(f"{further_analysis}\n\n")
                file.write("## Plots\n")
                for img in os.listdir(SETTINGS["RESULTS_DIR"]):
                    if img.endswith(".png"):
                        file.write(f"![{img}](./{img})\n")
                file.write("\n## Summary\n")
                file.write(f"{summary_story}")
            logging.info(f"README.md has been successfully created at {readme_path}.")
        except Exception as error:
            logging.error(f"Error generating README file: {error}")

    def process_dataset(self, filepath):
        try:
            file_encoding = self.get_file_encoding(filepath)
            dataframe_iterator = pd.read_csv(filepath, encoding=file_encoding, chunksize=SETTINGS["CHUNK_SIZE"])
            dataframe = pd.concat([chunk for chunk in dataframe_iterator])

            logging.info(f"Dataset loaded successfully with shape {dataframe.shape}.")
            
            # Create a results folder specific to the dataset name
            dataset_name = os.path.splitext(os.path.basename(filepath))[0]
            results_folder = os.path.join(SETTINGS["RESULTS_DIR"], dataset_name)
            os.makedirs(results_folder, exist_ok=True)
            SETTINGS["RESULTS_DIR"] = results_folder

            # Perform analysis
            dataframe = self.feature_engineering(dataframe)
            missing_data = self.review_missing_values(dataframe)
            correlations = self.evaluate_correlation(dataframe)
            outliers = self.identify_outliers(dataframe)
            clusters = self.perform_data_clustering(dataframe)
            self.explore_distributions(dataframe)

            # Generate the README file in the dataset-specific folder
            self.create_readme(dataframe, missing_data, correlations, outliers, clusters)
            logging.info(f"Analysis results have been saved to the folder: {results_folder}")
        except Exception as error:
            logging.error(f"Error processing dataset: {error}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("Usage: python script.py <path_to_csv_file>")
        sys.exit(1)
    filepath = sys.argv[1]
    analysis_tool = DataAnalysisTool()
    analysis_tool.process_dataset(filepath)

