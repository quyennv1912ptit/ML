import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from CARTClassifier import CARTClassifier

df = pd.read_csv("drug200.csv")

print(df)

le_Sex = LabelEncoder()
le_BP = LabelEncoder()
le_Cholesterol = LabelEncoder()
le_Drug = LabelEncoder()

df['Sex'] = le_Sex.fit_transform(df['Sex'])
df['BP'] = le_BP.fit_transform(df['BP'])
df['Cholesterol'] = le_Cholesterol.fit_transform(df['Cholesterol'])
df['Drug'] = le_Drug.fit_transform(df['Drug'])

X = df.drop(columns=['Drug']).values
y = df['Drug'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

tree = CARTClassifier(criterion='gini', max_depth=5)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

def classifier_metrics(y_true, y_pred, n_classes):
    acc = np.sum(y_true == y_pred) / len(y_true)
    precisions = []
    recalls = []
    f1_scores = []

    for c in range(n_classes):
        TP = np.sum((y_pred == c) & (y_true == c))
        FP = np.sum((y_pred == c) & (y_true != c))
        FN = np.sum((y_pred != c) & (y_true == c))

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)

    return acc, macro_precision, macro_recall, macro_f1

# print(le_Drug.inverse_transform(y_pred))

acc, prec, recall, f1_score = classifier_metrics(y_test, y_pred, len(np.unique(y)))
print("ACC: ", acc)
print("PREC: ", prec)
print("RECALL: ", recall)
print("F1_SCORE: ", f1_score)

feature_names = df.drop(columns='Drug').columns

importances = tree.feature_importances

plt.figure(figsize=(8, 5))
if importances is not None:
    plt.bar(feature_names, importances, color='skyblue', edgecolor='black')

plt.title('Feature Importances - CART Classifier', fontsize=14, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)

plt.tight_layout()
plt.show()

# ---------------------------------Nếu loại bỏ feature không quan trọng ---------------#
print("Nếu loại bỏ feature không quan trọng")
X = df.drop(columns=['Drug', 'Sex']).values
y = df['Drug'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

tree = CARTClassifier(criterion='gini', max_depth=5)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

# print(le_Drug.inverse_transform(y_pred))

acc, prec, recall, f1_score = classifier_metrics(y_test, y_pred, len(np.unique(y)))
print("ACC: ", acc)
print("PREC: ", prec)
print("RECALL: ", recall)
print("F1_SCORE: ", f1_score)

feature_names = df.drop(columns=['Drug', 'Sex']).columns

importances = tree.feature_importances

plt.figure(figsize=(8, 5))
if importances is not None:
    plt.bar(feature_names, importances, color='skyblue', edgecolor='black')

plt.title('Feature Importances - CART Classifier', fontsize=14, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)

plt.tight_layout()
plt.show()

#-------------------- Nếu xóa feature quan trọng --------------------#

print(" Nếu xóa feature quan trọng")
X = df.drop(columns=['Drug', 'Na_to_K']).values
y = df['Drug'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

tree = CARTClassifier(criterion='gini', max_depth=5)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

# print(le_Drug.inverse_transform(y_pred))

acc, prec, recall, f1_score = classifier_metrics(y_test, y_pred, len(np.unique(y)))
print("ACC: ", acc)
print("PREC: ", prec)
print("RECALL: ", recall)
print("F1_SCORE: ", f1_score)

feature_names = df.drop(columns=['Drug', 'Na_to_K']).columns

importances = tree.feature_importances

plt.figure(figsize=(8, 5))
if importances is not None:
    plt.bar(feature_names, importances, color='skyblue', edgecolor='black')

plt.title('Feature Importances - CART Classifier', fontsize=14, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)

plt.tight_layout()
plt.show()
