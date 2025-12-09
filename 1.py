#Предварительный анализ и очистка
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

df = pd.read_csv('wdbc.data', header=None, names=column_names)

# Поиск пропущенных значений
missing_values = df.isnull().sum()

# Вывод признаков, где есть хотя бы одно пропущенное значение
print("Пропущенные значения по столбцам:")
print(missing_values[missing_values > 0])

# Дополнительно: общее число пропущенных значений
print("\nОбщее количество пропущенных значений во всём датасете:", df.isnull().sum().sum())
