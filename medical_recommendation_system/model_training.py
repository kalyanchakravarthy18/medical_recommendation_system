#Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

#Importing data
files = {
    "workout": "workout_df.csv",
    "description": "description.csv",
    "diets": "diets.csv",
    "medications": "medications.csv",
    "precautions": "precautions_df.csv",
    "symptoms": "symtoms_df.csv",
}

dfs = {name: pd.read_csv(path) for name, path in files.items()}
data_summary = {name: df.head() for name, df in dfs.items()}

#Data Preprocessing
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

#Splitting, vectorizing and encoding the required data for training the ML model
label_encoder = LabelEncoder()
merged_df["Disease_Encoded"] = label_encoder.fit_transform(merged_df["Disease"])

X = merged_df["Combined_Symptoms"]
y = merged_df["Disease_Encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Training and testing the model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
# fig, ax = plt.subplots(figsize=(20, 20))
# disp.plot(cmap="coolwarm", xticks_rotation="vertical", ax=ax)
# plt.title("Confusion Matrix")
# plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Random Forest Model used: {accuracy * 100}")

#Saving the model, vectorizer and encoder
model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"
encoder_path = "encoder.pkl"

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
joblib.dump(label_encoder, encoder_path)
print("Saved the model, vectorizer and encoders successfully!")