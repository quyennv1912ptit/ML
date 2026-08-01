import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

df = pd.read_csv("data.csv")

df = df.dropna(axis=1, how='all')

df_X = df.drop(columns=["id", "diagnosis"])
df_y = df["diagnosis"]
df_y_le = LabelEncoder()
df_y_encoded = df_y_le.fit_transform(df_y)

print(df_X.head())

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y_encoded, test_size=0.2, random_state=45)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

K_values = range(1, 31)
train_errors = []
val_errors = []

for k in K_values:
    clf = KNeighborsClassifier(n_neighbors=k, p=2)
    cv_results = cross_validate(
        clf, X_train_scaled, y_train,
        cv=5, scoring='accuracy',
        return_train_score=True
    )
    train_errors.append(1 - cv_results['train_score'].mean())
    val_errors.append(1 - cv_results['test_score'].mean())

optimal_k = K_values[val_errors.index(min(val_errors))]
print(f"K tối ưu (5-fold CV trên tập train): {optimal_k}")

knn = KNeighborsClassifier(n_neighbors=optimal_k, p=2)

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("acc: ", acc)
print("prec: ", prec)
print("recall: ", recall)
print("f1: ", f1)

plt.figure(figsize=(10, 6))
plt.plot(K_values, train_errors, label='Train Error', marker='o', color='blue')
plt.plot(K_values, val_errors, label='Validation Error (CV)', marker='o', color='red')
plt.axvline(x=optimal_k, color='gray', linestyle='--', label=f'Optimal K = {optimal_k}')
plt.title('Train vs CV Error Curve - Tìm K tối ưu cho KNN')
plt.xlabel('K')
plt.ylabel('Error rate')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()