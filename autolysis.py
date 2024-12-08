import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
import chardet
import requests

# Configuration for AI Proxy API
SETTINGS = {
    "PROXY_ENDPOINT": "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
    "RESULTS_DIR": os.path.dirname(os.path.abspath(__file__)),
    "MAX_IMAGES": 3
}

image_counter = 0  # Global counter to track the number of images generated

def get_proxy_key():
    try:
        return os.environ["AIPROXY_TOKEN"]
    except KeyError:
        print("Error: AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)

def query_ai(prompt, context):
    proxy_key = get_proxy_key()
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
        print(f"AI Proxy communication error: {error}")
        sys.exit(1)

def save_plot(filename):
    global image_counter
    if image_counter >= SETTINGS["MAX_IMAGES"]:
        print("Image generation limit reached.")
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
        image_counter += 1
    except Exception as error:
        print(f"Error saving plot {filename}: {error}")

def get_file_encoding(filepath):
    try:
        with open(filepath, 'rb') as file:
            raw_data = file.read()
        detected_encoding = chardet.detect(raw_data)
        return detected_encoding['encoding']
    except Exception as error:
        print(f"Error detecting encoding for file {filepath}: {error}")
        sys.exit(1)

def review_missing_values(dataframe):
    missing_counts = dataframe.isnull().sum()
    missing_percentages = (missing_counts / len(dataframe)) * 100
    missing_summary = missing_percentages[missing_percentages > 0].sort_values(ascending=False)
    if not missing_summary.empty:
        plt.figure()
        missing_summary.plot(kind='bar', color='lightblue')
        plt.title("Columns with Missing Data (%)")
        save_plot("missing_data_report.png")
    return missing_summary

def evaluate_correlation(dataframe):
    numeric_data = dataframe.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        correlations = numeric_data.corr()
        plt.figure()
        sns.heatmap(correlations, annot=True, cmap="coolwarm")
        plt.title("Correlation Analysis")
        save_plot("correlation_matrix.png")
        return correlations
    return None

def identify_outliers(dataframe):
    numeric_data = dataframe.select_dtypes(include=[np.number])
    outlier_data = {}
    for column in numeric_data.columns:
        q1 = numeric_data[column].quantile(0.25)
        q3 = numeric_data[column].quantile(0.75)
        iqr = q3 - q1
        outlier_data[column] = numeric_data[(numeric_data[column] < q1 - 1.5 * iqr) | (numeric_data[column] > q3 + 1.5 * iqr)].shape[0]
    return outlier_data

def perform_data_clustering(dataframe):
    numeric_data = dataframe.select_dtypes(include=[np.number])
    if not numeric_data.empty and numeric_data.shape[1] > 1:
        cluster_model = KMeans(n_clusters=3, random_state=42)
        dataframe['Cluster_Group'] = cluster_model.fit_predict(numeric_data.fillna(0))
        plt.figure()
        sns.scatterplot(
            x=numeric_data.columns[0], y=numeric_data.columns[1],
            hue='Cluster_Group', data=dataframe, palette='viridis'
        )
        plt.title("Cluster Representation")
        save_plot("clusters.png")
        return dataframe['Cluster_Group'].value_counts()
    return None

def explore_distributions(dataframe):
    numeric_columns = dataframe.select_dtypes(include=[np.number])
    for column in numeric_columns.columns:
        if image_counter >= SETTINGS["MAX_IMAGES"]:
            break
        plt.figure()
        sns.histplot(dataframe[column], kde=True, bins=30, color="darkblue")
        plt.title(f"Data Distribution: {column}")
        save_plot(f"{column}_distribution.png")

def create_readme(dataframe, missing_data, correlations, outliers, clusters):
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
    summary_story = query_ai("Choose one of the following heart-touching topics randmoly: 1) Love and Sacrifice, 2) Overcoming Grief, 3) Acts of Kindness, 4) Women empowerment, or 5) Redemption. Based on the dataset insights, craft a well-written, continuous story that evokes deep emotions, showcases a meaningful journey of transformation or connection, and concludes with a hopeful and feel-good ending.",
    analysis_context)

    further_analysis = query_ai("Propose additional analysis steps based on the current insights.", analysis_context)
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
        print(f"README.md has been successfully created at {readme_path}.")
    except Exception as error:
        print(f"Error generating README file: {error}")

def process_dataset(filepath):
    try:
        file_encoding = get_file_encoding(filepath)
        dataframe = pd.read_csv(filepath, encoding=file_encoding)
        print(f"Dataset loaded: {filepath} ({dataframe.shape[0]} rows, {dataframe.shape[1]} columns)")
        missing_data_report = review_missing_values(dataframe)
        correlation_report = evaluate_correlation(dataframe)
        outliers_report = identify_outliers(dataframe)
        cluster_analysis_report = perform_data_clustering(dataframe)
        explore_distributions(dataframe)
        create_readme(dataframe, missing_data_report, correlation_report, outliers_report, cluster_analysis_report)
    except Exception as error:
        print(f"Error processing dataset: {error}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset_file>")
        sys.exit(1)
    dataset_file = sys.argv[1]
    if not os.path.exists(dataset_file):
        print(f"Error: File '{dataset_file}' does not exist.")
        sys.exit(1)
    process_dataset(dataset_file)
