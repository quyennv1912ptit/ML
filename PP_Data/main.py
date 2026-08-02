import re
import numpy as np
import pandas as pd
import category_encoders as ce

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("winemag-data-130k-v2.csv")

df = df.dropna(subset=['price'])

cols_to_drop = ["id", "Unnamed: 0", "taster_twitter_handle", "region_2"]
df = df.drop(columns=cols_to_drop, errors='ignore')

df['vintage_year'] = df['title'].str.extract(r'\b(19\d{2}|20\d{2})\b', expand=False).astype(float)
df['clean_desc'] = df['description'].str.lower().apply(lambda x: re.sub(r'[^a-z\s]', '', str(x)))

df = df.drop(columns=['title', 'description'])

cols_to_fill = ['country', 'province', 'region_1', 'variety', 'winery', 'designation', 'taster_name']
df[cols_to_fill] = df[cols_to_fill].fillna('Unknown')

X = df.drop(columns=['price'])
y = df['price']

y = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_year_median = X_train['vintage_year'].median()
X_train['vintage_year'] = X_train['vintage_year'].fillna(train_year_median).astype(int)
X_test['vintage_year'] = X_test['vintage_year'].fillna(train_year_median).astype(int)

scaler = StandardScaler()
X_train['points'] = scaler.fit_transform(X_train[['points']])
X_test['points'] = scaler.transform(X_test[['points']])

target_cols = ['province', 'region_1', 'variety', 'winery', 'designation']
target_encoder = ce.TargetEncoder(cols=target_cols, smoothing=10)

X_train[target_cols] = target_encoder.fit_transform(X_train[target_cols], y_train)
X_test[target_cols] = target_encoder.transform(X_test[target_cols])

X_train = pd.get_dummies(X_train, columns=['country', 'taster_name'], dtype=int)
X_test = pd.get_dummies(X_test, columns=['country', 'taster_name'], dtype=int)

X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

tfidf = TfidfVectorizer(stop_words='english', max_features=1000, dtype=float)

train_tfidf_matrix = tfidf.fit_transform(X_train['clean_desc'])
test_tfidf_matrix = tfidf.transform(X_test['clean_desc'])

feature_names = tfidf.get_feature_names_out()

train_tfidf_df = pd.DataFrame(
    train_tfidf_matrix.toarray(), 
    columns=[f"word_{word}" for word in feature_names],
    index=X_train.index
)

test_tfidf_df = pd.DataFrame(
    test_tfidf_matrix.toarray(), 
    columns=[f"word_{word}" for word in feature_names],
    index=X_test.index
)

X_train = pd.concat([X_train, train_tfidf_df], axis=1).drop(columns=['clean_desc'])
X_test = pd.concat([X_test, test_tfidf_df], axis=1).drop(columns=['clean_desc'])

print(X_train)

print("Kích thước X_train:", X_train.shape)
print("Kích thước X_test:", X_test.shape)

print("\nĐang huấn luyện Ridge Regression...")
ridge_model = Ridge(alpha=1.0, random_state=42)

ridge_model.fit(X_train, y_train)
print("Huấn luyện xong!")

y_pred_log = ridge_model.predict(X_test)

y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_log)

mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
r2 = r2_score(y_test, y_pred_log)

print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
print(f"R2 Score: {r2:.4f}")
print(f"MAE:      ${mae:.2f}")
print(f"RMSE:     ${rmse:.2f}")