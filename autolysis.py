# /// script
# dependencies = [
#   "matplotlib",
#   "scikit-learn",
#   "requests",
#   "pandas",
#   "numpy",
#   "scipy",
#   "pathlib",
#   "tenacity",
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import zscore
from pathlib import Path
import requests
import base64
from tenacity import retry, stop_after_attempt, wait_fixed

# Retry policy: Stop after 3 attempts and wait 2 seconds between retries.
default_retry = retry(stop=stop_after_attempt(3), wait=wait_fixed(2))

# Function to securely retrieve API key from environment for reusability
@default_retry
def get_api_key():
    """Get the API key."""
    return os.environ.get("AIPROXY_TOKEN")

# Create output directories
@default_retry
def create_directories(base_dirs):
    """Create directories for output storage."""
    for base_dir in base_dirs:
        Path(base_dir).mkdir(parents=True, exist_ok=True)

# Load CSV file
@default_retry
def load_csv(file_path):
    """Load a CSV file with UTF-8 encoding and error handling."""
    with open(file_path, encoding='utf-8', errors='ignore') as file:
        df = pd.read_csv(file)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Analyze missing values
@default_retry
def analyze_missing_values(df):
    """Analyze missing values in the dataset."""
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    print(f"Total missing values: {total_missing}")
    return missing_values

# Generate summary statistics
@default_retry
def summary_statistics(df):
    """Generate summary statistics for numeric and categorical data."""
    numeric_summary = df.describe(include=[np.number])
    categorical_summary = df.describe(include=[object])
    return numeric_summary, categorical_summary

# Correlation matrix
@default_retry
def correlation_analysis(df):
    """Perform correlation analysis for numeric data."""
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    return correlation_matrix

# Data distribution visualization
@default_retry
def visualize_data_distribution(df):
    """Visualize the distribution of numeric data."""
    numeric_df = df.select_dtypes(include=[np.number])
    for column in numeric_df.columns:
        plt.figure(figsize=(8, 4))
        plt.hist(numeric_df[column], bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

# Outlier detection
@default_retry
def detect_outliers(df):
    """Detect outliers in numeric data using Z-scores."""
    numeric_df = df.select_dtypes(include=[np.number])
    z_scores = np.abs(zscore(numeric_df))
    outliers = (z_scores > 3).sum(axis=0)
    return outliers

# Perform clustering
@default_retry
def perform_clustering(df, n_clusters=3):
    """Perform K-means clustering on numeric data."""
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(numeric_df)
    return clusters

# Visualize data
@default_retry
def visualize_data(df, output_dir):
    """Generate visualizations for the dataset."""
    # Correlation heatmap
    correlation_matrix = correlation_analysis(df)
    plt.figure(figsize=(8, 6))
    plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="none")
    plt.colorbar()
    plt.title("Correlation Matrix")
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    correlation_path = f"{output_dir}/correlation_matrix.png"
    plt.savefig(correlation_path, dpi=100)
    plt.close()

    # Missing values bar plot
    missing_values = analyze_missing_values(df)
    plt.figure(figsize=(8, 4))
    missing_values[missing_values > 0].plot(kind="bar", color="orange")
    plt.title("Missing Values by Column")
    missing_values_path = f"{output_dir}/missing_values.png"
    plt.savefig(missing_values_path, dpi=100)
    plt.close()

    return [correlation_path, missing_values_path]

@default_retry
def encode_image(image_path):
    """Encode image to base64 for LLM integration."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Interact with LLM using API request via proxy
@default_retry
def query_llm(prompt, api_key, images=None):
    """Query the LLM via API for narrative generation."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    if images:
        base64_image = encode_image(images[0])
        data["messages"][0]["content"] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ]

    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        headers=headers,
        json=data
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()

# Generate narrative
@default_retry
def generate_narrative(df, analysis_results, output_dir, image_paths):
    """Generate a narrative summary using the LLM."""
    prompt = (
        f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
        f"Key insights:\n"
        f"- Missing values: {analysis_results['missing_values'].sum()} total.\n"
        f"- Detected outliers in columns: {analysis_results['outliers'][analysis_results['outliers'] > 0].to_dict()}\n"
        f"- Clusters identified: {len(np.unique(analysis_results['clusters']))}.\n\n"
        """Generate a concise summary highlighting significant findings. 
        Discuss potential implications, suggest further steps, and reference visualizations where appropriate."""
    )

    api_key = get_api_key()
    narrative = query_llm(prompt, api_key, images=image_paths)

    if narrative:
        readme_path = os.path.join(os.getcwd(), "README.md")
        with open(readme_path, "w") as f:
            f.write("# Analysis Narrative\n\n")
            f.write(narrative)
            print(f"Narrative saved to {readme_path}")

# Main script
@default_retry
def main():
    """Main function for orchestrating the data analysis pipeline."""
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.getcwd()

    create_directories([output_dir])
    df = load_csv(file_path)

    numeric_summary, categorical_summary = summary_statistics(df)
    missing_values = analyze_missing_values(df)
    outliers = detect_outliers(df)
    clusters = perform_clustering(df)

    analysis_results = {
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "missing_values": missing_values,
        "outliers": outliers,
        "clusters": clusters,
    }

    image_paths = visualize_data(df, output_dir)
    generate_narrative(df, analysis_results, output_dir, image_paths)

if __name__ == "__main__":
    main()
