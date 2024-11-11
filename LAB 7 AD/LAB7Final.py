import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import StandardScaler;
from sklearn.ensemble import RandomForestRegressor, IsolationForest;
from sklearn.metrics import mean_absolute_error, mean_squared_error;
import statsmodels.api as sm;

data = pd.read_csv('motorbike_ambulance_calls.csv')

print("Data Info:\n", data.info())
print("\nMissing values:\n", data.isnull().sum())
print("\nData Preview:\n", data.head())

categorical_columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
for col in categorical_columns:
    data[col] = data[col].astype('category')

X = data.drop(columns=['cnt'])
y = data['cnt']

plt.figure(figsize=(12, 6))
sns.boxplot(x='hr', y='cnt', data=data)
plt.title('Hourly Distribution of Ambulance Calls')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='weathersit', y='cnt', data=data)
plt.title('Distribution of Calls by Weather Situation')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X[['temp', 'atemp', 'hum', 'windspeed']] = scaler.fit_transform(X[['temp', 'atemp', 'hum', 'windspeed']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Calls", color="blue")
plt.plot(y_pred, label="Predicted Calls", color="red")
plt.title("Actual vs Predicted Ambulance Calls")
plt.legend()
plt.show()

data['cnt_ts'] = data['cnt'].values
decomposition = sm.tsa.seasonal_decompose(data['cnt_ts'], period=24, model='additive')
fig = decomposition.plot()
plt.show()

isolation_forest = IsolationForest(contamination=0.05, random_state=42)
data['anomaly'] = isolation_forest.fit_predict(data[['cnt']])

anomalies = data[data['anomaly'] == -1]
plt.figure(figsize=(12, 6))
plt.plot(data['cnt'].values, label="Calls")
plt.scatter(anomalies.index, anomalies['cnt'], color='red', label="Anomaly")
plt.title("Anomaly Detection in Ambulance Calls")
plt.legend()
plt.show()

print("Anomalies detected:\n", anomalies[['date', 'cnt']])