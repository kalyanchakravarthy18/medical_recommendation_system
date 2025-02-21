import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
encoder = joblib.load("encoder.pkl")

data = pd.read_csv("merged_df.csv")
symptom_columns = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]
symptom_options = sorted(data[symptom_columns].stack().dropna().unique())

st.title("Disease Prediction App")
st.write("Select symptoms to predict the possible disease and get additional information.")

selected_symptoms = []
for i in range(4):
    symptom = st.selectbox(f"Select symptom {i+1}", options=["None"] + symptom_options, index=0)
    selected_symptoms.append(symptom)

selected_symptoms = [symptom for symptom in selected_symptoms if symptom != "None"]

if st.button("Examine Symptoms"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    
    else:
        combined_symptoms = " ".join(selected_symptoms)
        input_vec = vectorizer.transform([combined_symptoms])
        predicted_encoded_disease = model.predict(input_vec)[0]
        predicted_disease = encoder.inverse_transform([predicted_encoded_disease])[0]

        disease_info = data[data["Disease"] == predicted_disease].iloc[0]

        st.subheader(f"Possible Disease: {predicted_disease}")
        st.write("**Description:**")
        st.write(f"{disease_info['Description']}")
        st.write("**Medication:**")
        st.write(f"{disease_info["Medication"]}")
        st.write("**Diet:**")
        st.write(f"{disease_info['Diet']}")
        st.write("**Precautions:**")
        precautions = [disease_info[f"Precaution_{i}"] for i in range(1, 5) if not pd.isna(disease_info[f"Precaution_{i}"])]
        for i, precaution in enumerate(precautions, start=1):
            st.write(f"- {precaution}")
        
        st.write("**Workout:**")
        st.write(f"{disease_info['workout']}")