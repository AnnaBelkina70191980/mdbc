#Кросс-валидация
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

# Подготовка стратифицированной кросс-валидации
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Масштабированные данные
X_scaled = scaler.fit_transform(X)
y = df["Diagnosis"]

# Список моделей
from xgboost import XGBClassifier

models_cv = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Voting Classifier": VotingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('knn', KNeighborsClassifier()),
        ('svc', SVC(probability=True)),
        ('rf', RandomForestClassifier()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ], voting='soft'),
    "Stacking Classifier": StackingClassifier(
        estimators=[
            ('lr', LogisticRegression()),
            ('knn', KNeighborsClassifier()),
            ('svc', SVC(probability=True)),
            ('rf', RandomForestClassifier())
        ],
        final_estimator=LogisticRegression()
    )
}

# Оценка через cross_val_score
cv_results = []
for name, model in models_cv.items():
    acc = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    f1 = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
    cv_results.append((name, acc.mean(), f1.mean()))

# Вывод
cv_df = pd.DataFrame(cv_results, columns=["Model", "Mean Accuracy", "Mean F1-score"])
cv_df = cv_df.sort_values(by="Mean F1-score", ascending=False)
print(cv_df)

