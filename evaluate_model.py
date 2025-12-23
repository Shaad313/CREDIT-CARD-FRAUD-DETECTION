import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

model = joblib.load('models/fraud_xgboost_model.pkl')
scaler = joblib.load('models/scaler.pkl')

data = pd.read_csv("C:\\Users\\razat\\Desktop\\creditcard2.csv")
X = scaler.transform(data.drop('Class', axis=1))
y = data['Class']

y_prob = model.predict_proba(X)[:, 1]
print("ROC-AUC Score:", roc_auc_score(y, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y, model.predict(X)))
