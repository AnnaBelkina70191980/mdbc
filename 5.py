#Описательный анализ данных
import pandas as pd

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

df = pd.read_csv("wdbc.data", header=None, names=column_names)

# Удалим неинформативный ID
df = df.drop(columns=["ID"])

# Подсчёт нулевых значений по каждому признаку
zero_counts = (df == 0).sum()

# Вывод только тех признаков, где встречаются нули
print("Нулевые значения по признакам:")
print(zero_counts[zero_counts > 0])

# Общее количество нулей в датасете
total_zeros = (df == 0).sum().sum()
print(f"\nОбщее количество нулей в датасете: {total_zeros}")
