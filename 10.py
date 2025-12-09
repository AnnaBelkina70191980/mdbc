#Поиск аномалий
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
anomaly_labels = iso.fit_predict(X_scaled)

# -1 = аномалия, 1 = нормальное значение
df['Anomaly'] = anomaly_labels
anomalies = df[df['Anomaly'] == -1]
print(f"Найдено аномалий: {len(anomalies)}")
