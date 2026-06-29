# Breast Cancer and Insurance Clustering

## Project Overview

This project applies unsupervised machine learning techniques to two datasets:

1. A preprocessed breast cancer dataset.
2. A health-insurance dataset.

The goal is to group similar records into clusters. These clusters can help identify patterns among patients and insurance profiles without using a target variable during training.

This project is intended for educational purposes. The clusters are descriptive patterns in the data and should not be used as medical diagnoses or as the sole basis for insurance decisions.

## Techniques Used

The project uses three clustering methods:

* K-Means Clustering
* MeanShift Clustering
* DBSCAN Clustering

## Project Steps

### 1. Data Import

The project reads two CSV files:

* `data_refined.csv` for the breast cancer dataset
* `insurance.csv` for the insurance dataset

### 2. Preprocessing

The datasets are prepared before clustering by:

* Removing unnecessary columns such as ID and diagnosis columns
* Encoding categorical variables using one-hot encoding
* Handling missing values
* Scaling features using `StandardScaler`

Scaling is important because clustering algorithms use distances between data points. Features with larger numerical values could otherwise dominate the clustering process.

### 3. K-Means Clustering

The Elbow Method is used to calculate the Sum of Squared Errors for several values of K. The elbow curve helps select an appropriate number of clusters.

The project uses:

* Two clusters for the breast cancer dataset
* Three clusters for the insurance dataset

The K-Means results are visualized using Principal Component Analysis (PCA), which reduces the dataset to two dimensions for plotting.

### 4. MeanShift Clustering

MeanShift is applied to both datasets. Different bandwidth values are tested for the breast cancer dataset because bandwidth controls how the algorithm groups nearby observations.

Unlike K-Means, MeanShift does not require the number of clusters to be chosen in advance.

### 5. DBSCAN Clustering

DBSCAN is applied to the breast cancer dataset as an additional clustering technique.

DBSCAN identifies dense groups of data points and can label unusual observations as noise. Noise points receive a cluster label of `-1`.

## Output Files

The code saves the following files in an `outputs` folder:

* `breast cancer_elbow_curve.png`
* `insurance_elbow_curve.png`
* `breast_kmeans_clusters.png`
* `insurance_kmeans_clusters.png`
* `breast_meanshift_clusters.png`
* `insurance_meanshift_clusters.png`
* `breast_dbscan_clusters.png`
* `breast_cancer_clustered.csv`
* `insurance_clustered.csv`

## Requirements

Install the required Python libraries:

```bash
pip install pandas matplotlib scikit-learn
```

## How to Run

1. Place `data_refined.csv` and `insurance.csv` in the same folder as the Python file.
2. Run the following command:

```bash
python breast_cancer_clustering.py
```

3. Review the generated figures and CSV files in the `outputs` folder.

## Conclusion

K-Means, MeanShift, and DBSCAN provide different ways to identify groups within the same data.

K-Means is simple and useful when an approximate number of clusters is known. MeanShift can automatically find cluster centers based on density. DBSCAN is useful for identifying dense groups and possible outliers.

Comparing these techniques helps show that clustering results can depend on the chosen method, preprocessing decisions, and parameter values.
