#Понижение размерности
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['Diagnosis'], palette="coolwarm", alpha=0.7)
plt.title("t-SNE визуализация (по метке Diagnosis)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()
