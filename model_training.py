
 # Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importing data
files = {
    "workout": "workout_df.csv",
    "description": "description.csv", 
    "diets": "diets.csv",
    "medications": "medications.csv",
    "precautions": "precautions_df.csv",
    "symptoms": "symtoms_df.csv",
}

dfs = {name: pd.read_csv(path) for name, path in files.items()}

# Data Preprocessing
dfs["workout"].rename(columns={"disease": "Disease"}, inplace=True)
merged_df = dfs["symptoms"].merge(dfs["description"], on="Disease", how="left") \
    .merge(dfs["medications"], on="Disease", how="left") \
    .merge(dfs["diets"], on="Disease", how="left") \
    .merge(dfs["precautions"], on="Disease", how="left") \
    .merge(dfs["workout"], on="Disease", how="left")

req_cols = [
    "Symptom_1",
    "Symptom_2",
    "Symptom_3",
    "Symptom_4",
    "Disease",
    "Description",
    "Medication",
    "Diet",
    "Precaution_1",
    "Precaution_2",
    "Precaution_3",
    "Precaution_4",
    "workout",
]
merged_df = merged_df[req_cols]

merged_df["Combined_Symptoms"] = merged_df[["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]].apply(
    lambda x: " ".join(x.dropna()), axis=1
)

# Splitting, vectorizing, and encoding the required data
label_encoder = LabelEncoder()
merged_df["Disease_Encoded"] = label_encoder.fit_transform(merged_df["Disease"])

X = merged_df["Combined_Symptoms"]
y = merged_df["Disease_Encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Defining hybrid model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gx_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Voting Classifier
hybrid_model = VotingClassifier(estimators=[
    ('random_forest', rf_model),
    ('xgboost', gx_model),
    ('gradient_boosting', gb_model)
], voting='hard')

# Training hybrid model
hybrid_model.fit(X_train_vec, y_train)
y_pred = hybrid_model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy of the Hybrid Model: {accuracy * 100}")
print(f"Precision of the Hybrid Model: {precision * 100}")
print(f"Recall of the Hybrid Model: {recall * 100}")
f1 = f1_score(y_test, y_pred, average='weighted')  # Calculate F1 score
print(f"F1 Score of the Hybrid Model: {f1 * 100}")  # Print F1 score

# Sav ng the model, vectorizer, and encoder
model_path = "hybrid_model.pkl"
vectorizer_path = "vectorizer.pkl"
encoder_path = "encoder.pkl"

joblib.dump(hybrid_model, model_path)
joblib.dump(vectorizer, vectorizer_path)
joblib.dump(label_encoder, encoder_path)
print("Saved the hybrid model, vectorizer, and encoder successfully!")

#Accuracy of the Hybrid Model: 99.3798955613577
#Precision of the Hybrid Model: 99.39792753686592
#Recall of the Hybrid Model: 99.3798955613577 



