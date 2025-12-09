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
df = df.drop(columns=["ID"])

# Преобразуем Diagnosis
df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

# Функция для подсчета выбросов в признаке по IQR
def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).sum()

# Подсчёт выбросов по каждому признаку
outliers = df.drop(columns=['Diagnosis']).apply(count_outliers)

# Вывод признаков с наибольшим числом выбросов
print("Количество выбросов по каждому признаку:")
print(outliers.sort_values(ascending=False))

# Общее число выбросов
total_outliers = outliers.sum()
print(f"\nОбщее количество выбросов в датасете: {total_outliers}")

