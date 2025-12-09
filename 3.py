#Описательный анализ данных
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка датасета
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

# Очистка данных: удаление ID и кодирование Diagnosis
df = df.drop(columns=['ID'])
df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})

# Расчет корреляционной матрицы
corr_matrix = df.corr()

# Визуализация корреляционной матрицы
plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5, center=0)
plt.title("Корреляционная матрица признаков (WDBC Dataset)", fontsize=16)
plt.tight_layout()
plt.show()
