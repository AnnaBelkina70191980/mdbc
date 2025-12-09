#Описательный анализ данных
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
column_names = [
    "ID", "Diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

df = pd.read_csv('wdbc.data', header=None, names=column_names)

# Очистка
df = df.drop(columns=["ID"])
df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

# Гистограммы
plt.figure(figsize=(20, 25))
for i, column in enumerate(df.columns[1:], 1):  # Пропускаем Diagnosis
    plt.subplot(6, 5, i)
    sns.histplot(df[column], bins=30, kde=True, color='skyblue')
    plt.title(column, fontsize=10)
    plt.xlabel('')
    plt.ylabel('')
plt.tight_layout()
plt.suptitle("Гистограммы распределения признаков", fontsize=18, y=1.02)
plt.show()

