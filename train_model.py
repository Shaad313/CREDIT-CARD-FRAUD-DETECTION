import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load data
data = pd.read_csv("C:\\Users\\razat\\Desktop\\creditcard2.csv")
data.columns = data.columns.str.strip()  # remove spaces

# Auto-detect target column
target_col = None
for col in data.columns:
    if col.lower() in ["class", "fraud", "is_fraud", "target"]:
        target_col = col
        break

if target_col is None:
    raise ValueError("No fraud target column found!")

print("Target column detected:", target_col)

X = data.drop(target_col, axis=1)
y = data[target_col]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Handle imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train model
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train_res, y_train_res)

# Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Save model
joblib.dump(model, "models/fraud_xgboost_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
