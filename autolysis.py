import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import zscore
from pathlib import Path
import requests

# Set OpenAI API key and API base URL for proxy
api_key = os.environ.get("AIPROXY_TOKEN")

# Create output directories
def create_directories(base_dirs):
    """Create directories for output storage."""
    for base_dir in base_dirs:
        Path(base_dir).mkdir(parents=True, exist_ok=True)

# Load CSV file
def load_csv(file_path):
    """Load a CSV file with UTF-8 encoding and error handling."""
    try:
        with open(file_path, encoding='utf-8', errors='ignore') as file:
            df = pd.read_csv(file)
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

# Analyze missing values
def analyze_missing_values(df):
    """Analyze missing values in the dataset."""
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    print(f"Total missing values: {total_missing}")
    return missing_values

# Generate summary statistics
def summary_statistics(df):
    """Generate summary statistics for numeric and categorical data."""
    numeric_summary = df.describe(include=[np.number])
    categorical_summary = df.describe(include=[object])
    return numeric_summary, categorical_summary

# Correlation matrix
def correlation_analysis(df):
    """Perform correlation analysis for numeric data."""
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    return correlation_matrix

# Outlier detection
def detect_outliers(df):
    """Detect outliers in numeric data using Z-scores."""
    numeric_df = df.select_dtypes(include=[np.number])
    z_scores = np.abs(zscore(numeric_df))
    outliers = (z_scores > 3).sum(axis=0)
    return outliers

# Perform clustering
def perform_clustering(df, n_clusters=3):
    """Perform K-means clustering on numeric data."""
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(numeric_df)
    return clusters

# Visualize data
def visualize_data(df, output_dir):
    """Generate visualizations for the dataset."""
    sns.set(style="whitegrid")
    
    # Correlation heatmap
    correlation_matrix = correlation_analysis(df)
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
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

# Interact with LLM using API request via proxy
def query_llm(prompt, images=None):
    """Query the LLM via API for narrative generation."""
    try:
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
            data["images"] = images[:1]  # Send only the first image to save tokens

        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error querying LLM: {e}")
        return None

# Generate narrative
def generate_narrative(df, analysis_results, output_dir, image_paths):
    """Generate a narrative summary using the LLM."""
    prompt = (
        f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
        f"Key insights:\n"
        f"- Missing values: {analysis_results['missing_values'].sum()} total.\n"
        f"- Detected outliers in columns: {analysis_results['outliers'][analysis_results['outliers'] > 0].to_dict()}\n"
        f"- Clusters identified: {len(np.unique(analysis_results['clusters']))}.\n\n"
        "Generate a concise summary highlighting significant findings. Use Markdown formatting and suggest implications."
    )

    narrative = query_llm(prompt, images=image_paths)

    if narrative:
        with open(f"{output_dir}/README.md", "w") as f:
            f.write("# Analysis Narrative\n\n")
            f.write(narrative)
            print("Narrative saved.")

# Main script
def main():
    """Main function for orchestrating the data analysis pipeline."""
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = f"{base_name}"

    # Create directories
    create_directories([output_dir])

    # Load dataset
    df = load_csv(file_path)

    # Perform analysis
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

    # Visualize results
    image_paths = visualize_data(df, output_dir)

    # Generate narrative
    generate_narrative(df, analysis_results, output_dir, image_paths)

if __name__ == "__main__":
    main()
