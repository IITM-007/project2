# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "requests",
#   "ipykernel",  # Added ipykernel
# ]
# ///
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import requests
import json
import openai 

def extract_numerical_insights(dataset):
    """
    Perform comprehensive statistical extraction on numerical columns
    """
    print("Initiating statistical exploration...")  # Operational marker
    
    # Numerical column extraction
    numerical_columns = dataset.select_dtypes(include=[np.number])
    
    # Comprehensive statistical computation
    statistical_summary = numerical_columns.agg([
        'count', 
        'mean', 
        'median', 
        'std', 
        'min', 
        'max', 
        lambda x: x.quantile(0.25), 
        lambda x: x.quantile(0.75)
    ])
    
    # Missing value computation
    column_nullity = dataset.isnull().sum()
    
    print("Statistical extraction completed.")  # Operational marker
    return statistical_summary, column_nullity


def identify_statistical_anomalies(dataset):
    """
    Detect statistical outliers using interquartile range methodology
    """
    print("Commencing anomaly detection...")  # Operational marker
    
    # Select numerical domains
    numerical_domain = dataset.select_dtypes(include=[np.number])
    
    # Quartile-based outlier computation
    q1_values = numerical_domain.quantile(0.25)
    q3_values = numerical_domain.quantile(0.75)
    
    interquartile_range = q3_values - q1_values
    
    # Outlier boundary computation
    lower_bound = q1_values - 1.5 * interquartile_range
    upper_bound = q3_values + 1.5 * interquartile_range
    
    # Anomaly identification
    anomalies = ((numerical_domain < lower_bound) | (numerical_domain > upper_bound)).sum()
    
    print("Anomaly detection finalized.")  # Operational marker
    return anomalies


def generate_exploratory_visuals(correlation_matrix, anomalies, source_dataset, output_directory):
    """
    Create comprehensive visualization suite
    """
    print("Initiating visual representation generation...")  # Operational marker
    
    visualization_outputs = {}
    
    # Correlation matrix visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.7)
    plt.title('Interdimensional Correlation Matrix')
    correlation_path = os.path.join(output_directory, 'correlation_matrix.png')
    plt.savefig(correlation_path)
    plt.close()
    visualization_outputs['correlation'] = correlation_path
    
    # Anomaly visualization
    if not anomalies.empty and anomalies.sum() > 0:
        plt.figure(figsize=(12, 6))
        anomalies.plot(kind='bar', color='crimson')
        plt.title('Statistical Anomaly Landscape')
        plt.xlabel('Feature Domains')
        plt.ylabel('Anomaly Frequency')
        anomaly_path = os.path.join(output_directory, 'outliers.png')
        plt.savefig(anomaly_path)
        plt.close()
        visualization_outputs['anomalies'] = anomaly_path
    
    # Distribution visualization
    numerical_columns = source_dataset.select_dtypes(include=[np.number]).columns
    if len(numerical_columns) > 0:
        initial_numerical_column = numerical_columns[0]
        plt.figure(figsize=(12, 6))
        sns.histplot(source_dataset[initial_numerical_column], kde=True, color='navy', bins=35)
        plt.title(f'Distribution Landscape: {initial_numerical_column}')
        distribution_path = os.path.join(output_directory, 'distribution_.png')
        plt.savefig(distribution_path)
        plt.close()
        visualization_outputs['distribution'] = distribution_path
    
    print("Visual representation generation completed.")  # Operational marker
    return visualization_outputs


def create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir, narrative_text):
    """
    Create the README.md with a narrative and visualizations
    """
    print("Creating README file...")  # Debugging line
    readme_file = os.path.join(output_dir, 'README.md')
    try:
        with open(readme_file, 'w') as f:
            f.write("# Automated Data Analysis Report\n\n")

            # Introduction Section
            f.write("## Introduction\n")
            f.write("This is an automated analysis of the dataset, providing summary statistics, visualizations, and insights from the data.\n\n")

            # Summary Statistics Section
            f.write("## Summary Statistics\n")
            f.write("The summary statistics of the dataset are as follows:\n")
            f.write("\n| Statistic    | Value |\n")
            f.write("|--------------|-------|\n")

            # Write summary statistics for each column (mean, std, min, etc.)
            for column in summary_stats.columns:
                f.write(f"| {column} - Mean | {summary_stats.loc['mean', column]:.2f} |\n")
                f.write(f"| {column} - Std Dev | {summary_stats.loc['std', column]:.2f} |\n")
                f.write(f"| {column} - Min | {summary_stats.loc['min', column]:.2f} |\n")
                f.write(f"| {column} - 25th Percentile | {summary_stats.loc['25%', column]:.2f} |\n")
                f.write(f"| {column} - 50th Percentile (Median) | {summary_stats.loc['50%', column]:.2f} |\n")
                f.write(f"| {column} - 75th Percentile | {summary_stats.loc['75%', column]:.2f} |\n")
                f.write(f"| {column} - Max | {summary_stats.loc['max', column]:.2f} |\n")
                f.write("|--------------|-------|\n")
            
            f.write("\n")

            # Missing Values Section (Formatted as Table)
            f.write("## Missing Values\n")
            f.write("The following columns contain missing values, with their respective counts:\n")
            f.write("\n| Column       | Missing Values Count |\n")
            f.write("|--------------|----------------------|\n")
            for column, count in missing_values.items():
                f.write(f"| {column} | {count} |\n")
            f.write("\n")

            # Outliers Detection Section (Formatted as Table)
            f.write("## Outliers Detection\n")
            f.write("The following columns contain outliers detected using the IQR method (values beyond the typical range):\n")
            f.write("\n| Column       | Outlier Count |\n")
            f.write("|--------------|---------------|\n")
            for column, count in outliers.items():
                f.write(f"| {column} | {count} |\n")
            f.write("\n")

            # Correlation Matrix Section
            f.write("## Correlation Matrix\n")
            f.write("Below is the correlation matrix of numerical features, indicating relationships between different variables:\n\n")
            f.write("![Correlation Matrix](correlation_matrix.png)\n\n")

            # Outliers Visualization Section
            f.write("## Outliers Visualization\n")
            f.write("This chart visualizes the number of outliers detected in each column:\n\n")
            f.write("![Outliers](outliers.png)\n\n")

            # Distribution Plot Section
            f.write("## Distribution of Data\n")
            f.write("Below is the distribution plot of the first numerical column in the dataset:\n\n")
            f.write("![Distribution](distribution_.png)\n\n")

            # Conclusion Section
            f.write("## Conclusion\n")
            f.write("The analysis has provided insights into the dataset, including summary statistics, outlier detection, and correlations between key variables.\n")
            f.write("The generated visualizations and statistical insights can help in understanding the patterns and relationships in the data.\n\n")

            # Story Section
            f.write("## Data Story\n")
            f.write(narrative_text)

        print(f"README file created: {readme_file}")  # Debugging line
        return readme_file
    except Exception as e:
        print(f"Error writing to README.md: {e}")
        return None


def question_llm(prompt, context):
    """
    Generate a heart-touching narrative using the OpenAI API
    """
    print("Generating story using LLM...")  # Debugging line
    try:
        # Get the AIPROXY_TOKEN from the environment variable
        token = os.environ["AIPROXY_TOKEN"]

        # Set the custom API base URL for the proxy
        api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

        # Construct the full prompt with emphasis on emotional depth
        full_prompt = f"""
        Create a profoundly moving and emotionally resonant story that emerges from the data analysis. 
        The narrative should:
        - Reveal the human stories hidden behind the numbers
        - Evoke deep empathy and emotional connection
        - Transform cold statistics into a warm, compassionate narrative
        - Use metaphors and personal perspectives to humanize the data

        Data Context:
        {context}

        Storytelling Guidelines:
        - The story must touch the heart
        - Use poetic language and deep emotional insight
        - Connect abstract data points to real human experiences
        - Create a narrative that makes the reader feel deeply
        - Explore themes of hope, resilience, transformation, and human connection

        Story Prompt:
        {prompt}
        """

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        # Prepare the body with the model and prompt
        data = {
            "model": "gpt-4o-mini",  # Specific model for proxy
            "messages": [
                {"role": "system", "content": "You are a deeply empathetic storyteller who can transform data into an emotional journey."},
                {"role": "user", "content": full_prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.8
        }

        # Send the POST request to the proxy
        response = requests.post(api_url, headers=headers, data=json.dumps(data))

        # Check for successful response
        if response.status_code == 200:
            # Extract the story from the response
            story = response.json()['choices'][0]['message']['content'].strip()
            print("Story generated.")  # Debugging line
            return story
        else:
            print(f"Error with request: {response.status_code} - {response.text}")
            return """
## A Story of Resilience

In the quiet spaces between numbers and data points, there lies a profound human story. Our data is more than just statistics—it's a testament to the incredible journey of human experience.

Each number represents a heartbeat, a moment of struggle, a breath of hope. Behind every data point is a person, with dreams, challenges, and an unbreakable spirit of resilience.

Though our analysis reveals patterns and anomalies, the true richness lies in the untold stories of courage, adaptation, and transformation that these numbers hint at but cannot fully capture.

In the end, data is just a reflection of our shared human narrative—complex, beautiful, and endlessly inspiring.
"""

    except Exception as e:
        print(f"Error: {e}")
        return """
## A Story of Resilience

In the quiet spaces between numbers and data points, there lies a profound human story. Our data is more than just statistics—it's a testament to the incredible journey of human experience.

Each number represents a heartbeat, a moment of struggle, a breath of hope. Behind every data point is a person, with dreams, challenges, and an unbreakable spirit of resilience.

Though our analysis reveals patterns and anomalies, the true richness lies in the untold stories of courage, adaptation, and transformation that these numbers hint at but cannot fully capture.

In the end, data is just a reflection of our shared human narrative—complex, beautiful, and endlessly inspiring.
"""


def main(csv_file):
    """
    Main function that integrates all the steps
    """
    print("Starting the analysis...")  # Debugging line

    # Try reading the CSV file with 'ISO-8859-1' encoding to handle special characters
    try:
        df = pd.read_csv(csv_file, encoding='ISO-8859-1')
        print("Dataset loaded successfully!")  # Debugging line
    except UnicodeDecodeError as e:
        print(f"Error reading file: {e}")
        return

    summary_stats, missing_values = extract_numerical_insights(df)

    # Debugging print
    print("Summary Stats:")
    print(summary_stats)

    outliers = identify_statistical_anomalies(df)

    # Debugging print
    print("Outliers detected:")
    print(outliers)

    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    # Select only numeric columns for correlation
    corr_matrix = df.select_dtypes(include=[np.number]).corr()

    # Visualize the data and check output paths
    heatmap_file, outliers_file, dist_plot_file = generate_exploratory_visuals(corr_matrix, outliers, df, output_dir)

    print("Visualizations saved.")

    # Generate the story using the LLM
    story = question_llm("Generate a deeply emotional story from the analysis", 
                         context=f"Dataset Analysis:\nSummary Statistics:\n{summary_stats}\n\nMissing Values:\n{missing_values}\n\nCorrelation Matrix:\n{corr_matrix}\n\nOutliers:\n{outliers}")

    # Create the README file with the analysis and the story
    readme_file = create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir, story)
    
    print(f"Analysis complete! Results saved in '{output_dir}' directory.")
    print(f"README file: {readme_file}")
    print(f"Visualizations: {heatmap_file}, {outliers_file}, {dist_plot_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])
