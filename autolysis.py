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

# Configuration file or settings dictionary
SETTINGS = {
    "PROXY_ENDPOINT": "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
    "RESULTS_DIR": os.path.dirname(os.path.abspath(__file__)),
    "MAX_IMAGES": 3,
    "MAX_TOKENS": 1024,
    "MODEL": "gpt-4o-mini"
}

class DataAnalysisTool:
    def __init__(self):
        self.image_counter = 0

    def get_proxy_key(self):
        try:
            return os.environ["AIPROXY_TOKEN"]
        except KeyError:
            raise EnvironmentError("AIPROXY_TOKEN environment variable not set.")

    def query_ai(self, prompt, context):
        proxy_key = self.get_proxy_key()
        api_headers = {"Authorization": f"Bearer {proxy_key}", "Content-Type": "application/json"}
        request_body = {
            "model": SETTINGS["MODEL"],
            "max_tokens": SETTINGS["MAX_TOKENS"],
            "messages": [{"role": "user", "content": f"{prompt}\nContext:\n{context}"}]
        }
        response = requests.post(SETTINGS["PROXY_ENDPOINT"], headers=api_headers, json=request_body)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def save_plot(self, filename):
        if self.image_counter >= SETTINGS["MAX_IMAGES"]:
            print("Image generation limit reached.")
            return
        plt.gcf().set_size_inches(5.12, 5.12)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SETTINGS["RESULTS_DIR"], filename),
            bbox_inches='tight',
            dpi=100
        )
        plt.close()
        self.image_counter += 1

    def get_file_encoding(self, filepath):
        with open(filepath, 'rb') as file:
            raw_data = file.read()
        detected_encoding = chardet.detect(raw_data)
        return detected_encoding['encoding']

    def review_missing_values(self, dataframe):
        missing_summary = dataframe.isnull().mean() * 100
        missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
        if not missing_summary.empty:
            sns.barplot(x=missing_summary.index, y=missing_summary.values, color="lightblue")
            plt.xticks(rotation=45, ha='right')
            plt.title("Columns with Missing Data (%)")
            plt.xlabel("Columns")
            plt.ylabel("Missing Percentage")
            self.save_plot("missing_data_report.png")
        return missing_summary

    def evaluate_correlation(self, dataframe):
        numeric_data = dataframe.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return None
        correlations = numeric_data.corr()
        sns.heatmap(correlations, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Analysis")
        self.save_plot("correlation_matrix.png")
        return correlations

    def identify_outliers(self, dataframe):
        numeric_data = dataframe.select_dtypes(include=[np.number])
        outlier_counts = {}
        for column in numeric_data.columns:
            q1 = numeric_data[column].quantile(0.25)
            q3 = numeric_data[column].quantile(0.75)
            iqr = q3 - q1
            outliers = numeric_data[(numeric_data[column] < q1 - 1.5 * iqr) | (numeric_data[column] > q3 + 1.5 * iqr)]
            outlier_counts[column] = len(outliers)
        return outlier_counts

    def optimal_clusters(self, data, max_clusters=10):
        silhouette_scores = []
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data)
            score = silhouette_score(data, cluster_labels)
            silhouette_scores.append((n_clusters, score))
        optimal = max(silhouette_scores, key=lambda x: x[1])
        return optimal[0]

    def perform_data_clustering(self, dataframe):
        numeric_data = dataframe.select_dtypes(include=[np.number])
        if numeric_data.empty or numeric_data.shape[1] <= 1:
            return None
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(numeric_data.fillna(0))
        optimal_n_clusters = self.optimal_clusters(reduced_data)
        kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
        clusters = kmeans.fit_predict(reduced_data)
        dataframe['Cluster_Group'] = clusters
        sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=clusters, palette="viridis", s=50)
        plt.title(f"Cluster Representation (Clusters: {optimal_n_clusters})")
        self.save_plot("clusters.png")
        return dataframe['Cluster_Group'].value_counts()

    def explore_distributions(self, dataframe):
        numeric_columns = dataframe.select_dtypes(include=[np.number])
        for column in numeric_columns.columns:
            if self.image_counter >= SETTINGS["MAX_IMAGES"]:
                break
            sns.histplot(dataframe[column], kde=True, bins=30, color="darkblue")
            plt.title(f"Data Distribution: {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            self.save_plot(f"{column}_distribution.png")

    def create_readme(self, dataframe, missing_data, correlations, outliers, clusters):
        analysis_summary = f"""
        Total records: {dataframe.shape[0]}
        Total fields: {dataframe.shape[1]}
        Columns: {', '.join(dataframe.columns)}

        Missing Values:
        {missing_data.to_string() if not missing_data.empty else 'None'}

        Correlations:
        {correlations.to_string() if correlations is not None else 'None'}

        Outliers:
        {outliers}

        Cluster Analysis:
        {clusters.to_string() if clusters is not None else 'None'}
        """
        narrative = self.query_ai(
            "Write a meaningful story using dataset insights.", analysis_summary
        )
        suggestions = self.query_ai(
            "Suggest next steps for further analysis.", analysis_summary
        )
        with open(os.path.join(SETTINGS["RESULTS_DIR"], "README.md"), "w") as file:
            file.write(f"# Analysis Report\n\n## Summary\n{analysis_summary}\n\n")
            file.write(f"## Narrative\n{narrative}\n\n")
            file.write(f"## Suggestions\n{suggestions}\n")

    def process_dataset(self, filepath):
        file_encoding = self.get_file_encoding(filepath)
        dataframe = pd.read_csv(filepath, encoding=file_encoding)
        results_folder = os.path.join(SETTINGS["RESULTS_DIR"], os.path.splitext(os.path.basename(filepath))[0])
        os.makedirs(results_folder, exist_ok=True)
        SETTINGS["RESULTS_DIR"] = results_folder
        missing_data = self.review_missing_values(dataframe)
        correlations = self.evaluate_correlation(dataframe)
        outliers = self.identify_outliers(dataframe)
        clusters = self.perform_data_clustering(dataframe)
        self.explore_distributions(dataframe)
        self.create_readme(dataframe, missing_data, correlations, outliers, clusters)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_csv_file>")
        sys.exit(1)
    filepath = sys.argv[1]
    tool = DataAnalysisTool()
    tool.process_dataset(filepath)
