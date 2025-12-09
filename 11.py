#Визуализация аномалий
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка и подготовка данных
column_names = [
    "ID", "Diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

df = pd.read_csv("wdbc.data", header=None, names=column_names)
df = df.drop(columns=["ID"])
df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

# Масштабирование признаков
features = df.drop(columns=["Diagnosis"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Выявление аномалий с помощью Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
df["Anomaly"] = iso.fit_predict(X_scaled)

# Преобразование для визуализации (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# Визуализация
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, x="PCA1", y="PCA2",
    hue="Anomaly", palette={1: "lightblue", -1: "red"},
    style="Anomaly", markers={1: "o", -1: "X"},
    alpha=0.7
)
plt.title("Визуализация аномалий (Isolation Forest + PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Аномалия", labels=["Нормальные", "Аномалии"])
plt.grid(True)
plt.tight_layout()
plt.show()
