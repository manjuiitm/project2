# MIT License
# Copyright (c) 2024 Manju Thomas
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import sys
import subprocess
from dotenv import load_dotenv

# Function to install a package
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"Error installing package {package}: {e}")
        sys.exit(1)

# Ensure matplotlib is installed
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("matplotlib is not installed. Attempting to install it now...")
    install_package("matplotlib")
    import matplotlib.pyplot as plt

# Ensure seaborb is installed
try:
    import seaborn as sns
except ModuleNotFoundError:
    print("Seaborn is not installed. Attempting to install it now...")
    install_package("seaborn")
    import seaborn as sns

def load_csv(filename):
    try:
        data = pd.read_csv(filename, encoding='ISO-8859-1')
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)

def analyze_data(data):
    analysis = {
        'shape': data.shape,
        'columns': data.columns.tolist(),
        'missing_values': data.isnull().sum(),
        'summary_statistics': data.describe()
    }
    return analysis

def generate_charts(data, output_dir):
    charts = []
    plt.figure(figsize=(10, 7))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    missing_chart = os.path.join(output_dir, "missing_values.png")
    plt.savefig(missing_chart, format="png")
    charts.append(missing_chart)
    plt.close()

    numeric_data = data.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        plt.figure(figsize=(12, 8))
        correlation = numeric_data.corr()
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        correlation_chart = os.path.join(output_dir, "correlation_matrix.png")
        plt.savefig(correlation_chart, format="png")
        charts.append(correlation_chart)
        plt.close()

        # Generate a single boxplot for all numeric columns
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=numeric_data)
        plt.title("Boxplot of Numeric Features")
        boxplot_chart = os.path.join(output_dir, "boxplot.png")
        plt.savefig(boxplot_chart, format="png")
        charts.append(boxplot_chart)
        plt.close()

    return charts

def clustering_analysis(data, output_dir):
    numeric_data = data.select_dtypes(include=[np.number]).dropna()
    if numeric_data.shape[0] == 0:
        print("No numeric data available for clustering.")
        return None

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    numeric_data['Cluster'] = clusters

    linkage_matrix = linkage(scaled_data, method='ward')
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title("Dendrogram")
    dendrogram_chart = os.path.join(output_dir, "dendrogram.png")
    plt.savefig(dendrogram_chart, format="png")
    plt.close()

    clustering_results_path = os.path.join(output_dir, "clustering_results.csv")
    numeric_data.to_csv(clustering_results_path, index=False)
    return clustering_results_path, dendrogram_chart

def generate_readme(analysis, charts, clustering_results_path, output_dir):
    readme_content = "# Automated Data Analysis Report\n\n"
    readme_content += f"### Dataset Summary\nThe dataset contains {analysis['shape'][0]} rows and {analysis['shape'][1]} columns.\n"
    readme_content += f"The columns of the dataset are: {', '.join(analysis['columns'])}.\n"
    readme_content += "\n### Missing Values Analysis\n"
    readme_content += f"```\n{analysis['missing_values']}\n```\n"
    readme_content += "\n### Summary Statistics\n"
    readme_content += f"```\n{analysis['summary_statistics']}\n```\n"

    readme_content += "\n### Visualizations\nBelow are the generated visualizations:\n"
    for chart in charts:
        readme_content += f"![Chart]({chart})\n"

    if clustering_results_path:
        readme_content += "\n### Clustering Results\n"
        readme_content += f"The clustering results are saved in `{clustering_results_path}`.\n"

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    return readme_path


def describe_images(output_dir):
    descriptions = []
    for file in os.listdir(output_dir):
        if file.endswith(".png"):
            descriptions.append(f"{file} - Generated visualization for {file.split('.')[0].replace('_', ' ').capitalize()}.")
    return "\n".join(descriptions)

def evaluate_with_llm(readme_path, image_descriptions, api_key):
    with open(readme_path, 'r') as file:
        readme_content = file.read()

    url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert in data analysis and visualization. You are tasked with evaluating the documentation and visualizations generated for a dataset."},
            {"role": "user", "content": f"""
Evaluate the following data analysis report, which includes a detailed README, visualizations, and clustering results. Your task is to review both the text-based insights in the README as well as the insights derived from the visualizations described below.

### README Content:
{readme_content}

### Visualizations:
{image_descriptions}

Ensure that the evaluation covers the following:
- General quality and clarity of the documentation.
- Appropriateness and clarity of the generated visualizations (charts).
- Key insights or patterns revealed by the data and visualizations, including any clustering analysis.
- Any potential improvements to the analysis or visualizations.
            """}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error during evaluation: {e}")
        return None

def main():
    # Hardcoded API token
    load_dotenv()
    api_key = os.getenv('API_KEY')

    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]

    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist.")
        return

    # Extract the base name without the .csv extension and create output directory
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_dir = os.path.join(os.getcwd(), base_name)
    os.makedirs(output_dir, exist_ok=True)
    #output_dir = os.getcwd()

    print(f"Output directory created: {output_dir}")
    data = load_csv(filename)
    analysis = analyze_data(data)
    charts = generate_charts(data, output_dir)
    clustering_results_path, dendrogram_chart = clustering_analysis(data, output_dir)
    if dendrogram_chart:
        charts.append(dendrogram_chart)

    readme_path = generate_readme(analysis, charts, clustering_results_path, output_dir)

    image_descriptions = describe_images(output_dir)

    feedback = evaluate_with_llm(readme_path, image_descriptions, api_key)

    if feedback:
        print("\n### LLM Feedback:\n")
        print(feedback)

if __name__ == "__main__":
    main()
