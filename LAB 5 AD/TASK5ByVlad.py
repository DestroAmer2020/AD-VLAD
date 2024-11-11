# Імпорт необхідних бібліотек
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.model_selection import train_test_split, GridSearchCV;
from sklearn.preprocessing import StandardScaler;
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier;
from xgboost import XGBClassifier;
from sklearn.metrics import accuracy_score;

# Завантаження даних
url = "https://www.kaggle.com/datasets/pkdarabi/diabetes-dataset-with-18-features"
data = pd.read_csv(url)

# Розділення на ознаки та цільову змінну
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Розбиваємо на навчальну та тестову вибірки (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабування ознак
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Побудова трьох Boosting моделей
# AdaBoost
ada_model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
ada_model.fit(X_train, y_train)

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42)
xgb_model.fit(X_train, y_train)

# Оцінка моделей
for model, name in zip([ada_model, gb_model, xgb_model], ['AdaBoost', 'Gradient Boosting', 'XGBoost']):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")

# Тонке налаштування параметрів з використанням GridSearchCV
# Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
grid_search_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=3)
grid_search_gb.fit(X_train, y_train)
print("Gradient Boosting Best Params:", grid_search_gb.best_params_)

# XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 0.9]
}
grid_search_xgb = GridSearchCV(XGBClassifier(random_state=42), param_grid_xgb, cv=3)
grid_search_xgb.fit(X_train, y_train)
print("XGBoost Best Params:", grid_search_xgb.best_params_)

# Регуляризація
# Gradient Boosting з регуляризацією
gb_model_regularized = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                  random_state=42)
gb_model_regularized.fit(X_train, y_train)

# XGBoost з регуляризацією
xgb_model_regularized = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8,
                                      reg_alpha=0.1, reg_lambda=1.0, random_state=42)
xgb_model_regularized.fit(X_train, y_train)

# Порівняння моделей з регуляризацією
for model, name in zip([ada_model, gb_model, gb_model_regularized, xgb_model, xgb_model_regularized], 
                       ['AdaBoost', 'Gradient Boosting', 'Gradient Boosting (Reg)', 'XGBoost', 'XGBoost (Reg)']):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")

# Відображення важливості ознак для XGBoost
plt.figure(figsize=(10, 6))
plt.barh(np.arange(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
plt.yticks(np.arange(len(X.columns)), X.columns)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.show()

# Візуалізація залежності продуктивності від зміни кількості дерев у Gradient Boosting
estimators = [50, 100, 150, 200]
accuracy_scores = []

for n in estimators:
    model = GradientBoostingClassifier(n_estimators=n, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Графік залежності
sns.lineplot(x=estimators, y=accuracy_scores)
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy")
plt.title("Effect of n_estimators on Gradient Boosting Performance")
plt.show()