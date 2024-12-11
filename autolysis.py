import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Input model for API
class FileInput(BaseModel):
    filename: str

@app.post("/analyze/")
async def analyze_csv(file_input: FileInput):
    """
    Analyze the given CSV file and generate insights.
    """
    filename = file_input.filename
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail=f"File {filename} not found.")

# Function to load the CSV file with error handling for encoding issues
def load_csv(filename):
    try:
        data = pd.read_csv(filename, encoding='ISO-8859-1')  # Or try 'latin1' or 'cp1252'
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)

# Function for basic data analysis
def analyze_data(data):
    analysis = {}
    analysis['shape'] = data.shape
    analysis['columns'] = data.columns.tolist()
    analysis['info'] = data.info()
    analysis['missing_values'] = data.isnull().sum()
    analysis['summary_statistics'] = data.describe()
    return analysis

# Function to generate charts
def generate_charts(data, output_dir):
    charts = []
    # Missing values heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    missing_chart = os.path.join(output_dir, "missing_values.png")
    plt.savefig(missing_chart, format="png")
    charts.append(missing_chart)
    plt.close()

    # Correlation matrix (if numeric columns exist)
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

        # Boxplot (for numeric columns)
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=numeric_data)
        plt.title("Boxplot of Numeric Features")
        boxplot_chart = os.path.join(output_dir, "boxplot.png")
        plt.savefig(boxplot_chart, format="png")
        charts.append(boxplot_chart)
        plt.close()

    return charts

# Function to perform clustering analysis (KMeans)
def clustering_analysis(data):
    numeric_data = data.select_dtypes(include=[np.number]).dropna()
    if numeric_data.shape[0] == 0:
        print("No numeric data available for clustering.")
        return None

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    numeric_data['Cluster'] = clusters
    return numeric_data

# Function for hierarchical clustering
def hierarchical_clustering(data):
    numeric_data = data.select_dtypes(include=[np.number]).dropna()
    if numeric_data.shape[0] == 0:
        print("No numeric data available for hierarchical clustering.")
        return None

    linked = linkage(numeric_data, method='ward')
    
    plt.figure(figsize=(12, 8))
    dendrogram(linked)
    plt.title("Hierarchical Clustering Dendrogram")
    dendrogram_chart = os.path.join(os.getcwd(), "dendrogram.png")
    plt.savefig(dendrogram_chart, format="png")
    plt.close()

    return dendrogram_chart

# Time Series Analysis: Forecasting with Exponential Smoothing (Holt-Winters)
def time_series_analysis(data, time_col, value_col):
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.set_index(time_col)
    
    # Ensure there's no missing data
    data = data.dropna(subset=[value_col])

    # Apply Exponential Smoothing
    model = ExponentialSmoothing(data[value_col], trend="add", seasonal="add", seasonal_periods=12)
    fit = model.fit()

    # Forecast the next 12 periods
    forecast = fit.forecast(12)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data[value_col], label='Observed')
    plt.plot(forecast.index, forecast, label='Forecast', linestyle='--', color='red')
    plt.title(f"Time Series Forecasting for {value_col}")
    plt.legend(loc='best')
    time_series_chart = os.path.join(os.getcwd(), "time_series_forecast.png")
    plt.savefig(time_series_chart, format="png")
    plt.close()

    return time_series_chart

# Geographic Analysis: Aggregate by location (e.g., region, country)
def geographic_analysis(data, location_col, value_col):
    # Aggregate the data by the geographic location
    geographic_data = data.groupby(location_col).agg({value_col: 'sum'}).reset_index()

    # Plot the geographic analysis
    plt.figure(figsize=(10, 6))
    sns.barplot(x=location_col, y=value_col, data=geographic_data)
    plt.title(f"Geographic Distribution of {value_col}")
    geographic_chart = os.path.join(os.getcwd(), "geographic_analysis.png")
    plt.savefig(geographic_chart, format="png")
    plt.close()

    return geographic_chart

# Network Analysis: Create a graph of relationships and plot
def network_analysis(data, source_col, target_col):
    G = nx.from_pandas_edgelist(data, source=source_col, target=target_col)
    plt.figure(figsize=(12, 12))
    nx.draw(G, with_labels=True, node_size=500, node_color='skyblue', font_size=10)
    plt.title("Network Analysis")
    network_chart = os.path.join(os.getcwd(), "network_analysis.png")
    plt.savefig(network_chart, format="png")
    plt.close()

    return network_chart

# Function to suggest analysis based on data (simulating function call suggestion)
def suggest_function_calls(data):
    suggestions = []
    if data.select_dtypes(include=[np.number]).shape[1] > 0:
        suggestions.append("Perform clustering analysis.")
    if data.select_dtypes(include=[np.number]).shape[0] > 50:
        suggestions.append("Try hierarchical clustering for grouping observations.")
    if 'date' in data.columns:
        suggestions.append("Consider time series analysis for trends and forecasting.")
    if 'location' in data.columns:
        suggestions.append("Perform geographic analysis to understand regional patterns.")
    if 'source' in data.columns and 'target' in data.columns:
        suggestions.append("Analyze network relationships between entities.")
    return suggestions

# Function to generate the README
def generate_readme(analysis, charts, clustering_results, dendrogram_chart, time_series_chart, geographic_chart, network_chart, output_dir, data):
    readme_content = "# Automated Data Analysis Report\n\n"
    
    # Introduction to the report
    readme_content += "### Introduction\n"
    readme_content += "This report presents an automated analysis of the dataset provided, including statistical summaries, visualizations, clustering results, and insights.\n\n"

    # Dataset Summary
    readme_content += "### Dataset Summary\n"
    readme_content += f"The dataset contains {analysis['shape'][0]} rows and {analysis['shape'][1]} columns.\n"
    readme_content += f"The columns of the dataset are: {', '.join(analysis['columns'])}.\n"
    readme_content += "Below is the information about the dataset:\n"
    readme_content += f"```\n{analysis['info']}\n```\n\n"

    # Missing Values Analysis
    readme_content += "### Missing Values Analysis\n"
    readme_content += "The following table shows the count of missing values in each column:\n"
    readme_content += f"```\n{analysis['missing_values']}\n```\n"

    # Summary Statistics
    readme_content += "### Summary Statistics\n"
    readme_content += "Here are the summary statistics of the numeric columns in the dataset:\n"
    readme_content += f"```\n{analysis['summary_statistics']}\n```\n"

    # Insights from the analysis
    readme_content += "### Insights\n"
    readme_content += "From the initial analysis, the following insights were observed:\n"
    readme_content += "- There are some missing values in the dataset. Further analysis can be done to handle these missing values.\n"
    readme_content += "- The dataset contains various numeric columns, and correlations between some of them are noteworthy.\n"
    readme_content += "- Boxplots show the distribution and potential outliers in the dataset.\n"
    readme_content += "- Based on clustering, some patterns emerge that group similar data points.\n"
    readme_content += "- Hierarchical clustering also provides a dendrogram to identify hierarchical relationships.\n\n"

    # Suggested analyses
    readme_content += "### Suggested Further Analyses\n"
    suggestions = suggest_function_calls(data)
    if suggestions:
        readme_content += "The following analyses may provide further insights:\n"
        for suggestion in suggestions:
            readme_content += f"- {suggestion}\n"
    else:
        readme_content += "No additional analyses suggested at this point.\n\n"

    # Visualizations
    readme_content += "### Visualizations\n"
    readme_content += "Below are the visualizations that support the analysis:\n"
    for chart in charts:
        readme_content += f"![Chart]({chart})\n"

    # Clustering Results Visualization
    if clustering_results is not None:
        clustering_chart = os.path.join(output_dir, "clustering_results.png")
        clustering_results.plot(kind='scatter', x=clustering_results.columns[0], y=clustering_results.columns[1], c=clustering_results['Cluster'], cmap='viridis')
        plt.title("Clustering Results")
        plt.savefig(clustering_chart)
        readme_content += f"![Clustering Results]({clustering_chart})\n"

    # Dendrogram Chart
    if dendrogram_chart:
        readme_content += f"![Dendrogram]({dendrogram_chart})\n"
    
    # Time Series Chart
    if time_series_chart:
        readme_content += f"![Time Series Forecast]({time_series_chart})\n"

    # Geographic Chart
    if geographic_chart:
        readme_content += f"![Geographic Analysis]({geographic_chart})\n"
    
    # Network Chart
    if network_chart:
        readme_content += f"![Network Analysis]({network_chart})\n"

    # Save the README file
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"README and charts saved in {output_dir}")

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]

    if not os.path.exists(filename):
        print(f"Error: File {filename} does not exist.")
        return

    # Load and analyze data
    data = load_csv(filename)
    analysis = analyze_data(data)

    # Create output directory
    output_dir = os.getcwd()

    # Generate charts and save them as PNG
    charts = generate_charts(data, output_dir)

    # Perform clustering and hierarchical clustering
    clustering_results = clustering_analysis(data)
    dendrogram_chart = hierarchical_clustering(data)

    # Time Series, Geographic, and Network analysis
    time_series_chart = None
    geographic_chart = None
    network_chart = None
    if 'date' in data.columns and 'value' in data.columns:
        time_series_chart = time_series_analysis(data, 'date', 'value')
    if 'location' in data.columns and 'value' in data.columns:
        geographic_chart = geographic_analysis(data, 'location', 'value')
    if 'source' in data.columns and 'target' in data.columns:
        network_chart = network_analysis(data, 'source', 'target')

    # Generate the README with results and charts
    generate_readme(analysis, charts, clustering_results, dendrogram_chart, time_series_chart, geographic_chart, network_chart, output_dir, data)

    print("Analysis complete. Files saved in the current directory.")

# Run the main function
if __name__ == "__main__":
    main()
