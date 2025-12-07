# ==========================================
# Crop Yield Prediction Project (Final Clean)
# Regression + Ensemble Models + Graphs
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 1. LOAD DATA (CSV ONLY)
# --------------------------------------------
df = pd.read_csv("archive (1)/yield_df.csv")   # <-- THIS IS CORRECT

print("===== FIRST 5 ROWS =====")
print(df.head())
print("\n===== COLUMNS =====")
print(df.columns)


# 2. CLEANING
# --------------------------------------------
df = df.dropna().reset_index(drop=True)

# Remove unnamed columns if exist
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

target_col = "hg/ha_yield"

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Columns available: {df.columns}")

y = df[target_col]
X = df.drop(columns=[target_col])


# 3. FEATURE SELECTION
# --------------------------------------------
numeric_features = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp"
]

categorical_features = ["Area", "Item"]

numeric_features = [c for c in numeric_features if c in X.columns]
categorical_features = [c for c in categorical_features if c in X.columns]


# 4. GRAPH 1 — Distribution of yield
plt.figure()
plt.hist(y, bins=30)
plt.xlabel("Yield (hg/ha)")
plt.ylabel("Frequency")
plt.title("Distribution of Crop Yield")
plt.show()


# 5. TRAIN–TEST SPLIT
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 6. PREPROCESSOR
# --------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)


# 7. MODEL EVALUATION FUNCTION
# --------------------------------------------
def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R²  :", r2)

    return name, mae, rmse, r2, model


# 8. MODELS
# --------------------------------------------
models = [
    ("Linear Regression", LinearRegression()),
    ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=42)),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05))
]

results = []
for name, algo in models:
    pipe = Pipeline([("pre", preprocessor), ("model", algo)])
    results.append(evaluate_model(name, pipe))


# 9. COMPARISON GRAPH — R² Score
# --------------------------------------------
model_names = [r[0] for r in results]
r2_scores = [r[3] for r in results]

plt.figure()
plt.bar(model_names, r2_scores, color='skyblue')
plt.ylabel("R² Score")
plt.title("Model Comparison")
plt.show()


# 10. SCATTER GRAPH — avg_temp vs yield
# --------------------------------------------
if "avg_temp" in df.columns:
    plt.figure()
    plt.scatter(df["avg_temp"], y)
    plt.xlabel("Average Temperature")
    plt.ylabel("Yield (hg/ha)")
    plt.title("avg_temp vs Yield")
    plt.show()


# 11. FINAL MODEL (best one)
best = max(results, key=lambda x: x[3])
best_model = best[4]

print("\nBest Model:", best[0])
