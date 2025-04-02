import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import tempfile

# Set page config as the very first Streamlit command
st.set_page_config(
    page_title="MediScan - Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model and necessary files - after page config
@st.cache_resource
def load_model_files():
    model = joblib.load("hybrid_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    encoder = joblib.load("encoder.pkl")
    data = pd.read_csv("merged_df.csv")
    return model, vectorizer, encoder, data

model, vectorizer, encoder, data = load_model_files()

# Define path for data storage with Streamlit Cloud compatibility
def get_data_path():
    # Check if running on Streamlit Cloud
    if os.environ.get('STREAMLIT_SHARING') or os.environ.get('IS_STREAMLIT_CLOUD'):
        # Use the .streamlit folder which is writable on Streamlit Cloud
        base_dir = ".streamlit"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        return os.path.join(base_dir, "patient_records.csv")
    else:
        # For local development, use temp directory
        temp_dir = tempfile.gettempdir()
        return os.path.join(temp_dir, "patient_records.csv")

PATIENT_DATA_FILE = get_data_path()

# Function to save patient data to CSV with improved error handling and validation
def save_patient_data(patient_name, patient_age, selected_symptoms, predicted_disease, medications, diet, workout, precautions):
    try:
        # Use local time instead of UTC to match the patient's actual check-in time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a dataframe for the new record
        new_record = pd.DataFrame({
            'Timestamp': [timestamp],
            'Patient Name': [patient_name],
            'Patient Age': [patient_age],
            'Symptoms': [", ".join(selected_symptoms)],
            'Predicted Disease': [predicted_disease],
            'Medications': [medications],
            'Diet Recommendations': [diet],
            'Workout Recommendations': [workout],
            'Precautions': [precautions]
        })
        
        # Check if file exists and load it, or create a new DataFrame
        if os.path.exists(PATIENT_DATA_FILE):
            try:
                existing_data = pd.read_csv(PATIENT_DATA_FILE)
                # Validate the loaded data
                if len(existing_data) == 0 or not all(col in existing_data.columns for col in new_record.columns):
                    st.warning("Existing data file has incorrect format. Creating a new file.")
                    updated_data = new_record
                else:
                    updated_data = pd.concat([existing_data, new_record], ignore_index=True)
            except Exception as e:
                st.warning(f"Could not read existing data file: {str(e)}. Creating a new file.")
                updated_data = new_record
        else:
            updated_data = new_record
        
        # Save to CSV (primary storage)
        updated_data.to_csv(PATIENT_DATA_FILE, index=False)
        
        # Also save to Excel for better formatting (if openpyxl is installed)
        try:
            excel_file = PATIENT_DATA_FILE.replace('.csv', '.xlsx')
            updated_data.to_excel(excel_file, index=False, engine='openpyxl')
        except Exception as e:
            pass  # Silently fail if Excel export isn't available
        
        return True, updated_data
    except Exception as e:
        import traceback
        st.error(f"Error saving data: {str(e)}")
        st.error(traceback.format_exc())
        return False, None

# Extract symptom options
symptom_columns = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]
symptom_options = sorted(data[symptom_columns].stack().dropna().unique())

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1e88e5;
        --secondary-color: #4caf50;
        --accent-color: #ff5722;
        --text-color: #333333;
        --bg-light: #f5f7fa;
        --card-border: #e0e0e0;
    }
    
    /* Header and title styling */
    .main-header {
        background: linear-gradient(90deg, var(--primary-color), #1565c0);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--primary-color);
    }
    
    .result-card {
        background-color: white;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
        border-top: 5px solid var(--secondary-color);
    }
    
    .disease-title {
        background-color: var(--secondary-color);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 50px;
        display: inline-block;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    /* Section styling */
    .section-title {
        color: var(--primary-color);
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-color);
    }
    
    .info-label {
        font-weight: 600;
        color: var(--text-color);
    }
    
    /* Precaution list styling - IMPROVED TEXT VISIBILITY */
    .precaution-item {
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        background-color: #f0f4f8;
        border-radius: 5px;
        border-left: 3px solid var(--accent-color);
        color: #000000;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, var(--secondary-color), #2e7d32);
        border: none;
        font-weight: 600;
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        box-shadow: 0 4px 10px rgba(76, 175, 80, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(76, 175, 80, 0.4);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: var(--bg-light);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--card-border);
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Animation for the examine button */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .animate-pulse .stButton > button {
        animation: pulse 2s infinite;
    }
    
    /* Patient info form styling */
    .patient-info-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary-color);
    }
    
    /* Data table styling */
    .table-wrapper {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="main-header">
    <h1 class="main-title">MediScan</h1>
    <p class="subtitle">Advanced Disease Prediction & Health Recommendation System</p>
</div>
""", unsafe_allow_html=True)

# Create tabs for main functionality and patient records
tab1, tab2 = st.tabs(["Disease Prediction", "Patient Records"])

with tab1:
    # Create a two-column layout
    col1, col2 = st.columns([1, 3])

    # Sidebar with patient info and symptoms selection
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Patient information section
        st.markdown('<div class="section-title">👤 Patient Information</div>', unsafe_allow_html=True)
        patient_name = st.text_input("Full Name", placeholder="Enter patient's full name")
        patient_age = st.number_input("Age", min_value=0, max_value=120, step=1, value=30)
        
        st.markdown('<div class="section-title">🔍 Symptom Analysis</div>', unsafe_allow_html=True)
        st.write("Select symptoms below for analysis:")
        
        selected_symptoms = []
        for i in range(4):
            symptom = st.selectbox(
                f"Symptom {i+1}",
                options=["None"] + list(symptom_options),
                index=0,
                key=f"symptom_{i}"
            )
            if symptom != "None":
                selected_symptoms.append(symptom)
        
        # Add animation class to the button container
        st.markdown('<div class="animate-pulse">', unsafe_allow_html=True)
        examine_button = st.button("Examine Symptoms", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick stats
        st.markdown('<div class="section-title">System Stats</div>', unsafe_allow_html=True)
        st.markdown(f"**Database Records:** {len(data):,}")
        st.markdown(f"**Symptoms in Database:** {len(symptom_options):,}")
        st.markdown(f"**Diseases Covered:** {data['Disease'].nunique():,}")
        
        # Patient records info
        if os.path.exists(PATIENT_DATA_FILE):
            try:
                patient_records = pd.read_csv(PATIENT_DATA_FILE)
                st.markdown(f"**Patients in Database:** {len(patient_records):,}")
            except:
                st.markdown("**Patients in Database:** 0")
        else:
            st.markdown("**Patients in Database:** 0")
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Main content area
    with col2:
        if examine_button:
            if not patient_name:
                st.warning("⚠️ Please enter the patient's name.")
            elif not selected_symptoms:
                st.warning("⚠️ Please select at least one symptom for analysis.")
            else:
                # Add a spinner for processing effect
                with st.spinner("Analyzing symptoms... Please wait."):
                    combined_symptoms = " ".join(selected_symptoms)
                    input_vec = vectorizer.transform([combined_symptoms])
                    
                    # Predict disease using the hybrid model
                    predicted_encoded_disease = model.predict(input_vec)[0]
                    predicted_disease = encoder.inverse_transform([predicted_encoded_disease])[0]
                    
                    # Fetch disease details
                    disease_info = data[data["Disease"] == predicted_disease].iloc[0]
                
                # Display the results with enhanced styling
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                
                # Patient information summary
                st.markdown(f'<div class="section-title">Patient</div>', unsafe_allow_html=True)
                st.markdown(f"**Name:** {patient_name}")
                st.markdown(f"**Age:** {patient_age} years")
                st.markdown(f"**Reported Symptoms:** {', '.join(selected_symptoms)}")
                
                # Disease name with colored background
                st.markdown(f'<div class="disease-title">🩺 {predicted_disease}</div>', unsafe_allow_html=True)
                
                # Disease description
                st.markdown('<div class="section-title">Description</div>', unsafe_allow_html=True)
                st.markdown(f"{disease_info['Description']}")
                
                # Create tabs for different information categories
                result_tab1, result_tab2, result_tab3 = st.tabs(["Treatment", "Lifestyle", "Precautions"])
                
                with result_tab1:
                    st.markdown('<div class="info-label">Recommended Medication:</div>', unsafe_allow_html=True)
                    medications = disease_info['Medication']
                    st.write(medications)
                
                with result_tab2:
                    st.markdown('<div class="info-label">Dietary Recommendations:</div>', unsafe_allow_html=True)
                    diet = disease_info['Diet']
                    st.write(diet)
                    
                    st.markdown('<div class="info-label">Exercise Recommendations:</div>', unsafe_allow_html=True)
                    workout = disease_info['workout']
                    st.write(workout)
                
                with result_tab3:
                    st.markdown('<div class="info-label">Key Precautions:</div>', unsafe_allow_html=True)
                    
                    # Fixed precautions handling
                    precaution_columns = ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
                    
                    # Check if any precautions exist
                    precaution_list = []
                    has_precautions = False
                    for col in precaution_columns:
                        if col in disease_info.index and pd.notna(disease_info[col]) and disease_info[col].strip() != "":
                            has_precautions = True
                            precaution_list.append(disease_info[col])
                            # Using clearer text styling for precautions
                            st.markdown(f'<div class="precaution-item">✓ {disease_info[col]}</div>', unsafe_allow_html=True)
                    
                    if not has_precautions:
                        st.write("No specific precautions available for this condition.")
                        precaution_list = ["No specific precautions available"]
                
                # Save the patient data
                precautions_text = ", ".join(precaution_list)
                try:
                    save_success, updated_data = save_patient_data(
                        patient_name, 
                        patient_age, 
                        selected_symptoms, 
                        predicted_disease, 
                        medications, 
                        diet, 
                        workout, 
                        precautions_text
                    )
                    
                    if save_success:
                        st.success("Patient record saved successfully!")
                        # Show the data storage location
                        st.info(f"Patient data saved to: {PATIENT_DATA_FILE}")
                    else:
                        st.error("Failed to save patient record.")
                except Exception as e:
                    st.error(f"Could not save patient data: {str(e)}")
                    st.info("You can still view the results, but the record wasn't saved to the database.")
                
                # Disclaimer
                st.info("⚠️ **Medical Disclaimer**: This prediction is based on symptoms only and is not a substitute for professional medical advice. Please consult a healthcare professional for proper diagnosis and treatment.")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Welcome message when no symptoms have been examined
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("👋 Welcome to MediScan")
            st.write("""
            MediScan uses advanced machine learning to analyze your symptoms and provide 
            possible disease predictions along with health recommendations.

            **How to use this system:**
            1. Enter patient information (name and age)
            2. Select symptoms from the panel on the left
            3. Click "Examine Symptoms" to analyze
            4. Review the prediction and recommendations
            
            All patient data and results will be automatically saved to a CSV file for
            record-keeping and future reference.
            
            Remember, this system provides preliminary insights only and is not a replacement 
            for professional medical consultation.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Information cards
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### 🔬 How It Works")
                st.write("""
                MediScan uses a hybrid machine learning model that analyzes patterns 
                between symptoms and diseases from thousands of medical records. 
                The system identifies potential matches and provides 
                relevant health information.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_b:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### 📊 Patient Records")
                st.write("""
                • All patient information is securely stored
                • Records include patient details, symptoms, and predictions
                • View all stored records in the Patient Records tab
                • Data is organized chronologically for easy reference
                • Download options available for offline analysis
                """)
                st.markdown('</div>', unsafe_allow_html=True)

# Patient Records Tab
with tab2:
    st.markdown('<div class="section-title">📋 Patient Records Database</div>', unsafe_allow_html=True)
    
    if os.path.exists(PATIENT_DATA_FILE):
        try:
            patient_records = pd.read_csv(PATIENT_DATA_FILE)
            if len(patient_records) > 0:
                # Add search/filter functionality
                search_term = st.text_input("Search by patient name or disease:", placeholder="Type to search...")
                
                filtered_records = patient_records
                if search_term:
                    filtered_records = patient_records[
                        patient_records['Patient Name'].str.contains(search_term, case=False) | 
                        patient_records['Predicted Disease'].str.contains(search_term, case=False)
                    ]
                
                # Display the records
                st.markdown('<div class="table-wrapper">', unsafe_allow_html=True)
                st.dataframe(filtered_records, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Record stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Patients", len(patient_records))
                with col2:
                    unique_diseases = patient_records['Predicted Disease'].nunique()
                    st.metric("Unique Diseases", unique_diseases)
                with col3:
                    avg_age = round(patient_records['Patient Age'].mean(), 1)
                    st.metric("Average Age", avg_age)
                
                # Download buttons for records
                st.markdown('### Download Records')
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = patient_records.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv_data,
                        file_name="patient_records.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    try:
                        # Try to create Excel data for download
                        from io import BytesIO
                        buffer = BytesIO()
                        patient_records.to_excel(buffer, index=False, engine='openpyxl')
                        excel_data = buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download as Excel",
                            data=excel_data,
                            file_name="patient_records.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except:
                        st.info("Excel download requires openpyxl package. Install with: pip install openpyxl")
            else:
                st.info("No patient records found. Use the Disease Prediction tab to analyze symptoms and save records.")
        except Exception as e:
            st.error(f"Error loading patient records: {str(e)}")
            st.info("Try using the Disease Prediction tab to create new records.")
    else:
        st.info("No patient records file exists yet. Use the Disease Prediction tab to analyze symptoms and save records.")

# Footer
st.markdown("""
<div class="footer">
    <p>MediScan Disease Prediction System v2.0</p>
    <p>Developed by kalyan chakravarthy and his team</p>
    <p>© {}</p>
</div>
""".format(datetime.now().year), unsafe_allow_html=True)
