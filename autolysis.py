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
#   "ipykernel",
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
    return (
        visualization_outputs.get('correlation', ''), 
        visualization_outputs.get('anomalies', ''), 
        visualization_outputs.get('distribution', '')
    )


def create_readme(summary_stats, missing_values, corr_matrix, outliers, output_dir, story=None):
    """
    Create a comprehensive README with analysis insights and optional narrative
    """
    print("Creating README file...")
    readme_file = os.path.join(output_dir, 'README.md')
    try:
        with open(readme_file, 'w', encoding='utf-8') as f:
            # Title and Introduction
            f.write("# 📊 Data Journey: From Numbers to Narrative\n\n")
            
            # Optional Story Section
            if story:
                f.write("## 📖 The Human Story Behind the Data\n")
                f.write(f"{story}\n\n")
            
            f.write("## 🔍 Dataset Characteristics\n")
            f.write("### Key Details\n")
            f.write(f"- **Total Columns**: {len(summary_stats.columns)}\n")
            f.write(f"- **Numeric Columns**: {len(summary_stats.columns)}\n")
            f.write(f"- **Total Observations**: {summary_stats.loc['count'].max()}\n\n")

            # Analysis Methodology
            f.write("## 🧪 Analysis Methodology\n")
            f.write("The analysis employed several robust statistical techniques:\n")
            f.write("- Descriptive Statistics\n")
            f.write("- Missing Value Analysis\n")
            f.write("- Outlier Detection (Interquartile Range Method)\n")
            f.write("- Correlation Matrix Computation\n\n")

            # Key Insights
            f.write("## 💡 Key Insights\n")
            
            # Summary Statistics Insights
            f.write("### Statistical Highlights\n")
            for column in summary_stats.columns:
                f.write(f"**{column}**:\n")
                f.write(f"- *Mean*: {summary_stats.loc['mean', column]:.2f}\n")
                f.write(f"- *Standard Deviation*: {summary_stats.loc['std', column]:.2f}\n")
                f.write(f"- *Range*: {summary_stats.loc['min', column]:.2f} - {summary_stats.loc['max', column]:.2f}\n\n")

            # Missing Values Insights
            f.write("### Missing Data\n")
            missing_columns = missing_values[missing_values > 0]
            if not missing_columns.empty:
                f.write("Columns with missing values:\n")
                for col, count in missing_columns.items():
                    f.write(f"- **{col}**: {count} missing entries\n")
            else:
                f.write("*No missing values detected in the dataset.*\n\n")

            # Outliers Insights
            f.write("### Outlier Analysis\n")
            if outliers.sum() > 0:
                f.write("Columns with statistical outliers:\n")
                for col, count in outliers.items():
                    if count > 0:
                        f.write(f"- **{col}**: {count} outliers detected\n")
            else:
                f.write("*No significant outliers found in the dataset.*\n\n")

            # Visualizations Section
            f.write("## 📈 Visualizations\n")
            f.write("### Correlation Matrix\n")
            f.write("![Correlation Matrix](correlation_matrix.png)\n\n")
            
            if outliers.sum() > 0:
                f.write("### Outliers Distribution\n")
                f.write("![Outliers](outliers.png)\n\n")

            f.write("### Data Distribution\n")
            f.write("![Distribution](distribution_.png)\n\n")

            # Implications and Recommendations
            f.write("## 🚀 Implications and Recommendations\n")
            f.write("Based on our comprehensive analysis, consider the following:\n")
            f.write("1. **Data Quality**: Review columns with missing values or outliers\n")
            f.write("2. **Further Investigation**: Deep dive into statistically significant variations\n")
            f.write("3. **Potential Next Steps**: \n")
            f.write("   - Data cleaning and preprocessing\n")
            f.write("   - Advanced statistical modeling\n")
            f.write("   - Machine learning feature engineering\n\n")

            # Closing Note
            f.write("## 🌟 Final Thoughts\n")
            f.write("Data tells a story. Our analysis is just the beginning of understanding the deeper narrative within these numbers.\n")

        print(f"README file created: {readme_file}")
        return readme_file
    except Exception as e:
        print(f"Error writing to README.md: {e}")
        return None


def question_llm(prompt, context):
    """
    Generate an emotional narrative using the OpenAI API
    """
    print("Generating story using LLM...")
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
            print("Story generated.")
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
    print("Starting the analysis...")

    # Try reading the CSV file with 'ISO-8859-1' encoding to handle special characters
    try:
        df = pd.read_csv(csv_file, encoding='ISO-8859-1')
        print("Dataset loaded successfully!")
    except UnicodeDecodeError as e:
        print(f"Error reading file: {e}")
        return

    summary_stats, missing_values = extract_numerical_insights(df)
    outliers = identify_statistical_anomalies(df)

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
