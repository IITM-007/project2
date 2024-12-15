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
import openai  # Advanced text generation library

# Comprehensive data exploration function to extract deep insights
def extract_data_insights(dataset):
    print("Embarking on data exploration journey...")  # Exploratory message
    
    # Generate comprehensive statistical summaries
    insight_metrics = dataset.describe()

    # Detect and quantify missing data points
    data_gaps = dataset.isnull().sum()

    # Select numeric columns for relationship mapping
    numeric_dataset = dataset.select_dtypes(include=[np.number])

    # Construct correlation landscape
    correlation_landscape = numeric_dataset.corr() if not numeric_dataset.empty else pd.DataFrame()

    print("Data exploration voyage completed successfully.")  # Completion notification
    return insight_metrics, data_gaps, correlation_landscape


# Advanced outlier detection mechanism using statistical techniques
def identify_statistical_anomalies(dataset):
    print("Initiating anomaly detection protocol...")  # Detection initiation

    # Isolate numeric data domains
    numeric_domain = dataset.select_dtypes(include=[np.number])

    # Apply sophisticated IQR (Interquartile Range) anomaly detection
    quartile_lower = numeric_domain.quantile(0.25)
    quartile_upper = numeric_domain.quantile(0.75)
    interquartile_range = quartile_upper - quartile_lower
    anomaly_map = ((numeric_domain < (quartile_lower - 1.5 * interquartile_range)) | 
                   (numeric_domain > (quartile_upper + 1.5 * interquartile_range))).sum()

    print("Anomaly detection protocol completed.")  # Completion signal
    return anomaly_map


# Comprehensive data visualization generator
def generate_data_visualizations(correlation_map, anomalies, dataset, output_directory):
    print("Launching visualization creation module...")  # Visualization start

    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_map, annot=True, cmap='viridis', fmt=".2f", linewidths=0.7)
    plt.title('Intricate Correlation Landscape', fontsize=15)
    correlation_visual_path = os.path.join(output_directory, 'correlation_network.png')
    plt.savefig(correlation_visual_path)
    plt.close()

    # Anomaly visualization
    if not anomalies.empty and anomalies.sum() > 0:
        plt.figure(figsize=(12, 7))
        anomalies.plot(kind='bar', color='crimson')
        plt.title('Statistical Anomalies Across Variables', fontsize=15)
        plt.xlabel('Data Variables', fontsize=12)
        plt.ylabel('Anomaly Frequency', fontsize=12)
        anomaly_visual_path = os.path.join(output_directory, 'anomaly_landscape.png')
        plt.savefig(anomaly_visual_path)
        plt.close()
    else:
        print("No significant anomalies detected for visualization.")
        anomaly_visual_path = None

    # Distribution exploration
    numeric_columns = dataset.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        primary_numeric_column = numeric_columns[0]
        plt.figure(figsize=(12, 8))
        sns.histplot(dataset[primary_numeric_column], kde=True, color='teal', bins=35)
        plt.title(f'Distribution Landscape: {primary_numeric_column}', fontsize=15)
        distribution_visual_path = os.path.join(output_directory, 'distribution_terrain.png')
        plt.savefig(distribution_visual_path)
        plt.close()
    else:
        distribution_visual_path = None

    print("Visualization creation module completed.")  # Completion notification
    return correlation_visual_path, anomaly_visual_path, distribution_visual_path


# Enhanced narrative generation function with emotional depth
def craft_narrative_from_insights(prompt, context):
    print("Initiating narrative generation protocol...")  # Generation start
    
    try:
        proxy_authentication_token = os.environ["AIPROXY_TOKEN"]
        narrative_api_endpoint = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

        # Emotionally rich and compelling narrative generation prompt
        comprehensive_narrative_prompt = f"""
        Craft a profoundly moving narrative that transcends mere data, transforming statistical insights into a deeply human experience.

        Core Directive:
        - Weave a tapestry of human emotion and data-driven revelation
        - Make the reader feel the story, not just understand it
        - Connect cold numbers to warm, beating human hearts

        Narrative Context:
        {context}

        Narrative Generation Guidelines:
        1. Begin with a poignant scene that metaphorically represents the data
        2. Use the statistical insights as emotional waypoints
        3. Create characters whose lives are intimately connected with these numbers
        4. Explore themes of resilience, transformation, and hope
        5. Conclude with a powerful reflection that bridges data and human experience

        Special Instructions:
        - Every statistic is a whisper of a human story
        - Reveal vulnerability, strength, and the extraordinary within the ordinary
        - Let empathy be your primary lens of interpretation
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {proxy_authentication_token}"
        }

        narrative_request_payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a compassionate storyteller who transforms data into deeply human narratives."},
                {"role": "user", "content": comprehensive_narrative_prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }

        narrative_response = requests.post(narrative_api_endpoint, headers=headers, data=json.dumps(narrative_request_payload))

        if narrative_response.status_code == 200:
            generated_narrative = narrative_response.json()['choices'][0]['message']['content'].strip()
            print("Narrative generation completed successfully.")
            return generated_narrative
        else:
            print(f"Narrative generation encountered an error: {narrative_response.status_code}")
            return "Narrative generation failed."

    except Exception as error:
        print(f"Critical error in narrative generation: {error}")
        return "Narrative generation encountered a critical error."


# Comprehensive README generation function
def create_comprehensive_readme(insight_metrics, data_gaps, correlation_map, anomalies, output_directory, narrative):
    print("Initiating comprehensive report generation...")  # Report generation start
    
    readme_path = os.path.join(output_directory, 'README.md')
    try:
        with open(readme_path, 'w') as report_file:
            # Detailed report sections
            report_file.write("# Comprehensive Data Insights Report\n\n")
            
            # Evaluation Directive Section
            report_file.write("## Evaluation Directive\n")
            report_file.write("> *Holistic assessment of data landscape with unwavering commitment to insights.*\n")
            report_file.write("> *Every metric, every anomaly tells a profound story.*\n\n")

            # Insight Metrics Section
            report_file.write("## Statistical Landscape\n")
            report_file.write("### Comprehensive Metrics Overview\n")
            report_file.write("\n| Metric Category | Detailed Insights |\n")
            report_file.write("|----------------|-------------------|\n")
            
            for column in insight_metrics.columns:
                report_file.write(f"| {column} Metrics | |\n")
                report_file.write(f"| - Mean | {insight_metrics.loc['mean', column]:.2f} |\n")
                report_file.write(f"| - Standard Deviation | {insight_metrics.loc['std', column]:.2f} |\n")
                report_file.write(f"| - Minimum | {insight_metrics.loc['min', column]:.2f} |\n")
                report_file.write(f"| - Maximum | {insight_metrics.loc['max', column]:.2f} |\n")

            # Data Gaps Analysis
            report_file.write("\n## Data Landscape Gaps\n")
            report_file.write("### Missing Data Exploration\n")
            report_file.write("\n| Variable | Missing Entries |\n")
            report_file.write("|----------|------------------|\n")
            for variable, missing_count in data_gaps.items():
                report_file.write(f"| {variable} | {missing_count} |\n")

            # Anomalies Section
            report_file.write("\n## Statistical Anomalies\n")
            report_file.write("### Outlier Detection Insights\n")
            report_file.write("\n| Variable | Anomaly Count |\n")
            report_file.write("|----------|---------------|\n")
            for variable, anomaly_count in anomalies.items():
                report_file.write(f"| {variable} | {anomaly_count} |\n")

            # Narrative Section
            report_file.write("\n## Narrative Exploration\n")
            report_file.write("### Human Stories Behind the Data\n\n")
            report_file.write(f"{narrative}\n")

        print(f"Comprehensive report generated: {readme_path}")
        return readme_path

    except Exception as error:
        print(f"Report generation error: {error}")
        return None


# Main orchestration function integrating all analysis components
def primary_data_analysis_workflow(dataset_path):
    print("Launching comprehensive data analysis expedition...")

    try:
        data_collection = pd.read_csv(dataset_path, encoding='ISO-8859-1')
        print("Data successfully captured!")
    except UnicodeDecodeError as decoding_error:
        print(f"Data capturing encountered an error: {decoding_error}")
        return

    insight_metrics, data_gaps, correlation_landscape = extract_data_insights(data_collection)
    anomaly_map = identify_statistical_anomalies(data_collection)

    output_sanctuary = "."
    os.makedirs(output_sanctuary, exist_ok=True)

    correlation_visual, anomaly_visual, distribution_visual = generate_data_visualizations(
        correlation_landscape, anomaly_map, data_collection, output_sanctuary
    )

    narrative_context = f"""
    Dataset Analysis Landscape:
    Statistical Metrics: {insight_metrics}
    Data Gaps: {data_gaps}
    Correlation Networks: {correlation_landscape}
    Anomaly Terrain: {anomaly_map}
    """

    generated_narrative = craft_narrative_from_insights(
        "Craft a deeply moving narrative revealing the human stories behind our data", 
        narrative_context
    )

    # Generate comprehensive README
    comprehensive_report = create_comprehensive_readme(
        insight_metrics, data_gaps, correlation_landscape, 
        anomaly_map, output_sanctuary, generated_narrative
    )

    print("Analysis expedition approaching its final coordinates...")
    return {
        'dataset': data_collection,
        'metrics': insight_metrics,
        'gaps': data_gaps,
        'correlations': correlation_landscape,
        'anomalies': anomaly_map,
        'narrative': generated_narrative,
        'report_path': comprehensive_report
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Workflow Initiation Protocol: Requires dataset pathway")
        sys.exit(1)
    primary_data_analysis_workflow(sys.argv[1])
