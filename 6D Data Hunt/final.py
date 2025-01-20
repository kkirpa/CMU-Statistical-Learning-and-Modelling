# importing necessary modules for Spark, data manipulation, visualization, and clustering
from pyspark import SparkContext
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.ml.evaluation import ClusteringEvaluator
import plotly.express as px

# initializing SparkContext for running Spark locally and SparkSession for DataFrame operations
sc = SparkContext("local", "6D Data Hunt")
spark = SparkSession.builder.appName("NormalizationExample").getOrCreate()

# loading and parsing the dataset "space.dat"
# reading the file into an rdd where each line represents a data point
# parsing each line to convert data points into floats and remove extra commas or whitespace
data = sc.textFile("space.dat")
parsed_data = data.map(lambda line: [float(x.strip(',').strip()) for x in line.split() if x])

# creating a dataframe with dense vectors as feature columns for compatibility with spark ml functions
df = parsed_data.map(lambda x: (Vectors.dense(x),)).toDF(["features"])

# normalizing the data by standardizing it (mean-centered and unit variance) using the standardscaler
# standardscaler transforms data to have zero mean and unit variance, which is useful for pca and clustering as siggested in slides
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
# fitting the scaler model to calculate mean and standard deviation for the entire dataset
scaler_model = scaler.fit(df)
# transforming the original dataset to add a new column, "scaledFeatures", which holds the normalized data
normalized_df = scaler_model.transform(df)

# applying pca to reduce data dimensionality from 6D to 3D for visualization and clustering
# reducing to 3 principal components will make it easier to visualize and work with the data in subsequent/later steps
pca = PCA(k=3, inputCol="scaledFeatures", outputCol="pcaFeatures")
# fitting the PCA model on the normalized data to identify principal components
pca_model = pca.fit(normalized_df)
# transforming the data to include a new column, "pcaFeatures", containing the 3D pca-transformed features
pca_result = pca_model.transform(normalized_df)

# # Clustering analysis with K-means
# kmeans = KMeans().setK(6).setSeed(1).setFeaturesCol("pcaFeatures")
# kmeans_model = kmeans.fit(pca_result)
# predictions = kmeans_model.transform(pca_result)

# using k-means clustering to identify clusters in the data
# setting a range of values for k (the number of clusters) to determine the optimal k using the silhouette score
wssse = []  # initializing list to store silhouette scores for each value of K
k_values = range(2, 11)  # defining range of K values to test, from 2 to 10
for k in k_values:
    # setting up k-means with the current k value and specifying to use "pcaFeatures" as input
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("pcaFeatures")
    # fitting the K-means model on PCA-transformed data for each value of K
    model = kmeans.fit(pca_result)
    # generating cluster predictions by transforming the pca-transformed data using the k-means model
    predictions = model.transform(pca_result)
    
    # evaluating cluster quality using silhouette score with respect to PCA-transformed features
    evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="pcaFeatures", metricName="silhouette")
    silhouette = evaluator.evaluate(predictions)
    # appending silhouette score to list for the corresponding K value
    wssse.append(silhouette)

# # plotting silhouette scores for each tested K value to visualize the Elbow Method
# plt.figure(figsize=(10, 6))
# plt.plot(k_values, wssse, marker='o')
# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("Silhouette Score")
# plt.title("Elbow Method for Optimal K")
# plt.savefig("elbow_method_plot.png")
# plt.close()

# determining the optimal K by selecting the K value with the highest silhouette score
optimal_k = k_values[np.argmax(wssse)]
# print(f"Optimal K determined by Elbow Method: {optimal_k}")

# performing clustering analysis again with the optimal number of clusters, K
kmeans = KMeans().setK(optimal_k).setSeed(1).setFeaturesCol("pcaFeatures")
# fitting the K-means model with optimal K value on PCA-transformed data
kmeans_model = kmeans.fit(pca_result)
# generating final predictions by transforming PCA data with optimal K-means model
predictions = kmeans_model.transform(pca_result)

# selecting relevant columns to include original, normalized, and PCA features, along with cluster prediction
data_with_clusters = predictions.select("features", "scaledFeatures", "pcaFeatures", "prediction")

# creating directory to store cluster PCA output summaries
output_dir = "cluster_pca_outputs"
os.makedirs(output_dir, exist_ok=True)

# getting unique cluster labels from predictions for individual cluster analysis
unique_labels = predictions.select("prediction").distinct().rdd.flatMap(lambda x: x).collect()
# print(f"Unique clusters found: {unique_labels}")

# initializing a list to store details about each cluster, such as dimensions, size, and shape
cluster_table = []

# initializing summary output file to store details of each cluster
summary_output = "object_summary.txt"
with open(summary_output, "w") as f_summary:
    # iterating through each unique cluster to analyze separately
    for cluster_id in unique_labels:
        # filtering data for the current cluster based on prediction column
        cluster_data = data_with_clusters.filter(data_with_clusters.prediction == cluster_id)
        
        # converting normalized and original feature vectors to arrays for extracting individual dimensions
        cluster_data = cluster_data.withColumn("scaledFeaturesArray", vector_to_array(col("scaledFeatures")))
        cluster_data = cluster_data.withColumn("originalFeaturesArray", vector_to_array(col("features")))

        # splitting each array into separate columns for each dimension (6 dimensions in total)
        for i in range(6):
            cluster_data = cluster_data.withColumn(f"scaledFeature_{i}", col("scaledFeaturesArray")[i])
            cluster_data = cluster_data.withColumn(f"originalFeature_{i}", col("originalFeaturesArray")[i])

        # calculating the mean for each dimension in normalized data to determine the cluster center in normalized space
        means_normalized = [f"avg(scaledFeature_{i})" for i in range(6)]
        cluster_center_normalized = cluster_data.selectExpr(*means_normalized).collect()[0]
        cluster_center_normalized = [round(c, 2) for c in cluster_center_normalized]  # rounding for readability

        # calculating the mean for each dimension in original data to determine the cluster center in original space
        means_original = [f"avg(originalFeature_{i})" for i in range(6)]
        cluster_center_original = cluster_data.selectExpr(*means_original).collect()[0]
        cluster_center_original = [round(c, 2) for c in cluster_center_original]        
        
        # determining the size of the cluster by counting the points in this cluster
        cluster_size = cluster_data.count()
        # print(f"Cluster {cluster_id} - Location (mean, normalized): {cluster_center_normalized}, Location (mean, original): {cluster_center_original} - Points: {cluster_size}")

        # performing pca on each cluster to identify primary dimensions (those with significant variance)
        cluster_pca = PCA(k=6, inputCol="scaledFeatures", outputCol="clusterPCAFeatures")
        cluster_pca_model = cluster_pca.fit(cluster_data)
        explained_variance = np.array(cluster_pca_model.explainedVariance)  # converting explained variance to NumPy array

        # counting primary dimensions based on explained variance threshold (>0.1) to determine dimensionality of the cluster
        primary_dimensions = np.sum(explained_variance > 0.1)
        # print(f"Cluster {cluster_id} - Primary Dimensions: {primary_dimensions}")
        
        # calculating range for each dimension within the cluster
        # for i in range(6):
        #     dim_min = cluster_data.agg(F.min(F.col(f"scaledFeature_{i}"))).collect()[0][0]
        #     dim_max = cluster_data.agg(F.max(F.col(f"scaledFeature_{i}"))).collect()[0][0]
        #     ranges.append(round(dim_max - dim_min, 2))

        # calculating range for each dimension within the cluster using min and max values in the original data
        ranges = []
        for i in range(6):
            dim_min = cluster_data.agg(F.min(F.col(f"originalFeature_{i}"))).collect()[0][0]
            dim_max = cluster_data.agg(F.max(F.col(f"originalFeature_{i}"))).collect()[0][0]
            ranges.append(round(dim_max - dim_min, 2))

        # calculating an estimated "size" for each dimension using percentile ranges
        size = sum(ranges)
        # this is to see the total spread of the cluster across all directions, in another case 
        # to calculate the maximum extent of this cluster in any given direction i could do this: 
        # size = max(ranges)
        
        # estimating shape based on primary dimensions and range consistency across dimensions
        if primary_dimensions == 1:
            shape = "Line"
        elif primary_dimensions == 2:
            shape = "Plane or Disk"
        elif primary_dimensions == 3:
            if np.allclose(ranges[:3], ranges[0], rtol=0.2):  # checking if ranges are approximately equal
                shape = "Sphere or Cube/Cuboid"
            else:
                shape = "Ellipsoid or Rectangular Prism"
        else:
            shape = "Higher-dimensional shape"

        # print(f"Cluster {cluster_id} - Estimated Shape: {shape}, Ranges: {ranges}")

        # saving cluster summary details in the cluster table to print to the console
        cluster_table.append(
            (f"Cluster {cluster_id}", f"{primary_dimensions}D", shape, ranges, size, cluster_center_original, cluster_center_normalized, "NA/not attempted", cluster_size)
        )

        # defining column names for the cluster summary table and displaying the summary data in a structured format
        columns = ["Object", "Dimension", "Estimated Shape", "Ranges in each Dim.", "Size/Spr", "Location (original)", "Location (normalized)", "Orientation", "Points"]
        cluster_table_df = spark.createDataFrame(cluster_table, columns)
        cluster_table_df.show(truncate=False)
        
        # writing the detailed information of each cluster to the summary file
        with open(summary_output, "a") as f_summary:
            f_summary.write(f"Object (Cluster {cluster_id}):\n")
            f_summary.write(f"Location (mean, normalized): {cluster_center_normalized}\n")
            f_summary.write(f"Location (mean, original): {cluster_center_original}\n")
            f_summary.write(f"Size (variance per dimension): {explained_variance}\n")
            f_summary.write(f"Primary Dimensions: {primary_dimensions}\n")
            f_summary.write(f"Shape: {shape}\n")
            f_summary.write(f"Dimension Ranges: {ranges}\n")
            f_summary.write(f"Points: {cluster_size}\n\n")

# creating directories for saving 2D and 3D cluster plots
output_dir_2d = "2d_cluster_plots"
os.makedirs(output_dir_2d, exist_ok=True)
for cluster_id in unique_labels:
    # filtering data for the current cluster and selecting PCA features for plotting in 2D
    cluster_data = predictions.filter(predictions.prediction == cluster_id).select("pcaFeatures")
    
    # extracting 2D PCA component coordinates for plotting each cluster in 2D space
    pca_x, pca_y = zip(*[(row[0][0], row[0][1]) for row in cluster_data.collect()])

    # setting up 2D scatter plot for the current cluster
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_x, pca_y, label=f"Cluster {cluster_id}", marker='o', s=20)

    # customizing plot appearance and saving as an image
    plt.title(f"2D PCA Plot of Cluster {cluster_id}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(os.path.join(output_dir_2d, f"Cluster_{cluster_id}_2D_PCA_Plot.png"))
    plt.close()

# creating directory to store 3D cluster plots
output_dir_3d = "3d_cluster_plots"
os.makedirs(output_dir_3d, exist_ok=True)
for cluster_id in unique_labels:
    # filtering data for the current cluster and selecting PCA features for plotting in 3D
    cluster_data = predictions.filter(predictions.prediction == cluster_id).select("pcaFeatures")
    
    # extracting 3D PCA component coordinates for plotting each cluster in 3D space
    pca_x, pca_y, pca_z = zip(*[(row[0][0], row[0][1], row[0][2]) for row in cluster_data.collect()])

    # setting up 3D scatter plot for the current cluster
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_x, pca_y, pca_z, label=f"Cluster {cluster_id}", marker='o', s=20)

    # customizing plot appearance and saving as an image
    ax.set_title(f"3D PCA Plot of Cluster {cluster_id}")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_zlabel("PCA Component 3")
    ax.legend()
    plt.savefig(os.path.join(output_dir_3d, f"Cluster_{cluster_id}_3D_PCA_Plot.png"))
    plt.close(fig)

# creating directory to store 3D cluster plots as interactive HTML files to observe the clusters further
output_dir_3d_html = "3d_cluster_plots_html"
os.makedirs(output_dir_3d_html, exist_ok=True)

for cluster_id in unique_labels:
    # filtering data for the current cluster and selecting PCA features for plotting in 3D
    cluster_data = predictions.filter(predictions.prediction == cluster_id).select("pcaFeatures")
    
    # extracting 3D PCA component coordinates for Plotly 3D scatter plot
    pca_x, pca_y, pca_z = zip(*[(row[0][0], row[0][1], row[0][2]) for row in cluster_data.collect()])

    # creating interactive 3D scatter plot using Plotly
    fig = px.scatter_3d(x=pca_x, y=pca_y, z=pca_z, opacity=0.7)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(title=f"3D PCA Plot of Cluster {cluster_id}")

    # saving the interactive plot as an HTML file
    fig.write_html(os.path.join(output_dir_3d_html, f"Cluster_{cluster_id}_3D_PCA_Plot.html"))

print(f"Interactive 3D plots saved in {output_dir_3d_html}.")

print("Clustered 2D and 3D plots saved in respective directories.")
print(f"Object summaries saved to {summary_output}")
