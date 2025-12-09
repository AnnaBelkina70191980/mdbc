#Представление результатов моделирования
from sklearn.ensemble import VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# === Повторим подготовку признаков ===
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Обучение ансамблей ===

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_scaled, y_train)
gb_pred = gb.predict(X_test_scaled)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_scaled, y_train)
xgb_pred = xgb.predict(X_test_scaled)

# VotingClassifier (Soft Voting)
voting = VotingClassifier(estimators=[
    ('lr', LogisticRegression()),
    ('knn', KNeighborsClassifier()),
    ('svc', SVC(probability=True)),
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
], voting='soft')
voting.fit(X_train_scaled, y_train)
voting_pred = voting.predict(X_test_scaled)

# StackingClassifier
stacking = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('knn', KNeighborsClassifier()),
        ('svc', SVC(probability=True)),
        ('rf', RandomForestClassifier())
    ],
    final_estimator=LogisticRegression()
)
stacking.fit(X_train_scaled, y_train)
stacking_pred = stacking.predict(X_test_scaled)

# === Оценка качества ===
models_preds = {
    "Random Forest": rf_pred,
    "Gradient Boosting": gb_pred,
    "XGBoost": xgb_pred,
    "Voting Classifier": voting_pred,
    "Stacking Classifier": stacking_pred
}

results_ensemble = []
for name, pred in models_preds.items():
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    results_ensemble.append((name, acc, f1))

results_ensemble_df = pd.DataFrame(results_ensemble, columns=["Model", "Accuracy", "F1-score"]).sort_values(by="F1-score", ascending=False)
print(results_ensemble_df)
