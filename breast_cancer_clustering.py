import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

breast = pd.read_csv("data_refined.csv")
insurance = pd.read_csv("insurance.csv")

def prepare_data(df, columns_to_drop):
    data = df.drop(columns=columns_to_drop, errors="ignore").copy()
    data = pd.get_dummies(data, drop_first=True)
    data = data.fillna(data.median())
    scaler = StandardScaler()
    return scaler.fit_transform(data), data

X_breast, breast_clean = prepare_data(
    breast,
    ["id", "ID", "diagnosis", "Diagnosed", "Unnamed: 32"]
)

X_insurance, insurance_clean = prepare_data(
    insurance,
    ["id", "ID"]
)

def elbow_plot(X, name):
    sse = []

    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        sse.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 9), sse, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squared Errors")
    plt.title(f"Elbow Curve: {name}")
    plt.xticks(range(2, 9))
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/{name.lower()}_elbow_curve.png", dpi=300)
    plt.show()

def plot_clusters(X, labels, title, filename):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=35)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/{filename}.png", dpi=300)
    plt.show()

elbow_plot(X_breast, "Breast Cancer")
elbow_plot(X_insurance, "Insurance")

kmeans_breast = KMeans(n_clusters=2, random_state=42, n_init=10)
breast_kmeans_labels = kmeans_breast.fit_predict(X_breast)

kmeans_insurance = KMeans(n_clusters=3, random_state=42, n_init=10)
insurance_kmeans_labels = kmeans_insurance.fit_predict(X_insurance)

plot_clusters(
    X_breast,
    breast_kmeans_labels,
    "K-Means Clusters: Breast Cancer Dataset",
    "breast_kmeans_clusters"
)

plot_clusters(
    X_insurance,
    insurance_kmeans_labels,
    "K-Means Clusters: Insurance Dataset",
    "insurance_kmeans_clusters"
)

breast_bandwidth = estimate_bandwidth(
    X_breast,
    quantile=0.20,
    n_samples=min(500, len(X_breast))
)

insurance_bandwidth = estimate_bandwidth(
    X_insurance,
    quantile=0.20,
    n_samples=min(500, len(X_insurance))
)

for multiplier in [0.75, 1.0, 1.25]:
    model = MeanShift(
        bandwidth=breast_bandwidth * multiplier,
        bin_seeding=True
    )

    labels = model.fit_predict(X_breast)

    print(
        "Breast Cancer MeanShift bandwidth:",
        round(breast_bandwidth * multiplier, 3),
        "| Clusters:",
        len(set(labels))
    )

mean_shift_breast = MeanShift(
    bandwidth=breast_bandwidth,
    bin_seeding=True
)

breast_mean_shift_labels = mean_shift_breast.fit_predict(X_breast)

mean_shift_insurance = MeanShift(
    bandwidth=insurance_bandwidth,
    bin_seeding=True
)

insurance_mean_shift_labels = mean_shift_insurance.fit_predict(X_insurance)

plot_clusters(
    X_breast,
    breast_mean_shift_labels,
    "MeanShift Clusters: Breast Cancer Dataset",
    "breast_meanshift_clusters"
)

plot_clusters(
    X_insurance,
    insurance_mean_shift_labels,
    "MeanShift Clusters: Insurance Dataset",
    "insurance_meanshift_clusters"
)

dbscan = DBSCAN(eps=1.5, min_samples=5)
breast_dbscan_labels = dbscan.fit_predict(X_breast)

plot_clusters(
    X_breast,
    breast_dbscan_labels,
    "DBSCAN Clusters: Breast Cancer Dataset",
    "breast_dbscan_clusters"
)

breast["KMeans_Cluster"] = breast_kmeans_labels
breast["MeanShift_Cluster"] = breast_mean_shift_labels
breast["DBSCAN_Cluster"] = breast_dbscan_labels

insurance["KMeans_Cluster"] = insurance_kmeans_labels
insurance["MeanShift_Cluster"] = insurance_mean_shift_labels

breast.to_csv(
    f"{OUTPUT_FOLDER}/breast_cancer_clustered.csv",
    index=False
)

insurance.to_csv(
    f"{OUTPUT_FOLDER}/insurance_clustered.csv",
    index=False
)

print("Breast Cancer K-Means clusters:", len(set(breast_kmeans_labels)))
print("Insurance K-Means clusters:", len(set(insurance_kmeans_labels)))
print("DBSCAN clusters:", len(set(breast_dbscan_labels)) - (1 if -1 in breast_dbscan_labels else 0))
print("DBSCAN noise points:", list(breast_dbscan_labels).count(-1))
print("Project completed successfully.")
