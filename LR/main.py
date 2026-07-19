from linear_regression import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("USA_Housing.csv")

print(df)

if 'Address' in df.columns:
    df = df.drop(columns=['Address'])

df = df.dropna()

X = df.drop(columns=['Price']).values
y = df['Price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

X_train = (X_train - X_mean) / (X_std + 1e-8)
X_test = (X_test - X_mean) / (X_std + 1e-8)

model = LinearRegression()

model.fit_gradient_descent(X_train, y_train)
print("W: ", model.w)

y_pred = model.predict(X_test)
print("Y_pred (20 mẫu đầu):", y_pred[:20])

mse = model.mse(y_test, y_pred)
print("MSE:", mse)
# Với những căn nhà giá từ 1- 2 triệu $, sai số khoảng 100k$ (5-10%)
print("RMSE:", np.sqrt(mse))

r_square = model.r_square(y_test, y_pred)
print("R^2: ", r_square)
