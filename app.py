import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Try to import XGBoost, show a friendly error if not installed
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.error("⚠️ XGBoost is not installed. Please run: pip install xgboost==1.7.6")
    st.stop()

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Healthcare Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the features used in the model - Order must match exactly with the model's expected order
FEATURES = [
    'payer_name', 'age', 'gender', 'race', 'ethnicity', 'income', 
    'encounterclass', 'base_encounter_cost', 'num_procedures', 
    'total_proc_base_cost', 'num_prior_conditions'
]
TARGET = 'out_of_pocket'

# Load the trained model
try:
    model = joblib.load('xgboost_out_of_pocket_model.pkl')
    # Convert to CPU if needed
    if hasattr(model, 'get_booster'):
        try:
            model.get_booster().set_param({'predictor': 'cpu_predictor'})
        except:
            pass  # Ignore if setting parameter fails
    st.sidebar.success('✅ Model loaded successfully')
except Exception as e:
    st.sidebar.error('❌ Error loading model: Make sure xgboost_out_of_pocket_model.pkl is in the same directory')
    st.sidebar.error(f'Error details: {str(e)}')
    model = None

# Label Encodings
PAYER_ENCODING = {
    'Aetna': 0,
    'Anthem': 1,
    'Blue Cross Blue Shield': 2,
    'Cigna Health': 3,
    'Dual Eligible': 4,
    'Humana': 5,
    'Medicaid': 6,
    'Medicare': 7,
    'NO_INSURANCE': 8,
    'UnitedHealthcare': 9
}

GENDER_ENCODING = {'F': 0, 'M': 1}

RACE_ENCODING = {
    'asian': 0,
    'black': 1,
    'hawaiian': 2,
    'native': 3,
    'other': 4,
    'white': 5
}

ETHNICITY_ENCODING = {
    'hispanic': 0,
    'nonhispanic': 1
}

ENCOUNTER_ENCODING = {
    'ambulatory': 0,
    'emergency': 1,
    'home': 2,
    'inpatient': 3,
    'outpatient': 4,
    'urgentcare': 5,
    'virtual': 6,
    'wellness': 7
}

# Custom CSS to make the app look more professional
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0083B8;
        color: white;
        border-radius: 5px;
        height: 3em;
        margin-top: 2em;
    }
    .stSelectbox {
        margin-bottom: 1em;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 100%;
    }
    div.block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    .insight-card {
        background-color: #f8f9fa;
        border-left: 4px solid #0083B8;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0 5px 5px 0;
    }
    .custom-header {
        color: #1E88E5;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E3F2FD;
    }
    .subheader {
        color: #0277BD;
        font-size: 1.5rem;
        font-weight: 500;
        margin: 1rem 0;
    }
    .info-text {
        color: #424242;
        font-size: 1rem;
        line-height: 1.6;
    }
    .highlight-box {
        background-color: #E3F2FD;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        border: 1px solid #BBDEFB;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0277BD;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #616161;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .sidebar .stRadio > label {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for user type selection
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital-2.png", width=100)
    st.markdown('<p class="custom-header" style="font-size: 1.5rem;">User Dashboard</p>', unsafe_allow_html=True)
    user_type = st.radio("Select User Type", ["Patient", "Insurance Provider"], key="user_type_radio")
    
    st.markdown("---")
    if user_type == "Insurance Provider":
        st.markdown('<p class="subheader">Provider Analytics</p>', unsafe_allow_html=True)
        st.markdown('<div class="highlight-box">Access detailed analytics and risk assessment for potential clients.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="subheader">Patient Portal</p>', unsafe_allow_html=True)
        st.markdown('<div class="highlight-box">Find the best insurance plan based on your health profile.</div>', unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="custom-header">🏥 Healthcare Cost Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
    <div class="highlight-box">
        <h4 style='margin-bottom: 0.5em; color: #0277BD;'>Out-of-Pocket Cost Prediction</h4>
        <p class="info-text">Enter your information to predict expected out-of-pocket healthcare costs.</p>
    </div>
""", unsafe_allow_html=True)

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Demographics")
    age = st.number_input("Age", min_value=0, max_value=120, value=30, key="age_input")
    gender = st.selectbox("Gender", ["F", "M"], key="gender_select")
    race = st.selectbox("Race", list(RACE_ENCODING.keys()), format_func=lambda x: x.capitalize(), key="race_select")
    ethnicity = st.selectbox("Ethnicity", list(ETHNICITY_ENCODING.keys()), format_func=lambda x: x.capitalize(), key="ethnicity_select")
    income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000, key="income_input")
    payer_name = st.selectbox("Insurance Provider", list(PAYER_ENCODING.keys()), key="payer_select")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Encounter Details")
    encounterclass = st.selectbox("Encounter Class", list(ENCOUNTER_ENCODING.keys()), format_func=lambda x: x.capitalize(), key="encounter_select")
    base_encounter_cost = st.number_input("Base Encounter Cost ($)", min_value=0.0, value=100.0, step=100.0, key="base_cost_input")
    num_procedures = st.number_input("Number of Procedures", min_value=0, value=0, key="procedures_input")
    total_proc_base_cost = st.number_input("Total Procedure Base Cost ($)", min_value=0.0, value=0.0, step=100.0, key="proc_cost_input")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Medical History")
    num_prior_conditions = st.number_input("Number of Prior Conditions", min_value=0, value=0, key="conditions_input")
    st.markdown("""
        <div style="padding: 1rem; background-color: #f8f9fa; border-radius: 5px; margin-top: 1rem;">
            <p style="color: #666666; margin-bottom: 0;">
                ℹ️ The number of prior conditions helps in assessing overall health status and potential cost implications.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Predict button
if st.button("Predict Out-of-Pocket Cost"):
    if model is None:
        st.error("⚠️ Model not loaded. Please ensure xgboost_out_of_pocket_model.pkl is available.")
    else:
        try:
            # Create input data dictionary with all features
            input_dict = {
                'payer_name': PAYER_ENCODING[payer_name],
                'age': age,
                'gender': GENDER_ENCODING[gender],
                'race': RACE_ENCODING[race],
                'ethnicity': ETHNICITY_ENCODING[ethnicity],
                'income': income,
                'encounterclass': ENCOUNTER_ENCODING[encounterclass],
                'base_encounter_cost': float(base_encounter_cost),
                'num_procedures': int(num_procedures),
                'total_proc_base_cost': float(total_proc_base_cost),
                'num_prior_conditions': int(num_prior_conditions)
            }
            
            # Create DataFrame with exact feature order
            input_data = pd.DataFrame([input_dict])[FEATURES]
            
            # Debug information
            st.sidebar.write("Input Data Preview:")
            st.sidebar.write(input_data)
            st.sidebar.write("Feature Order:", list(input_data.columns))
            
            # Make prediction
            predicted_cost = model.predict(input_data)[0]
            
            # Ensure non-negative prediction
            predicted_cost = max(0, predicted_cost)

            # Display Results
            st.markdown('<h2 class="subheader">Cost Analysis Results</h2>', unsafe_allow_html=True)
            
            # Create two columns for visualizations
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Cost Prediction Gauge
                max_range = max(predicted_cost * 2, 5000)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=predicted_cost,
                    number={'prefix': "$", 'valueformat': ",.2f"},
                    title={'text': "Estimated Out-of-Pocket Cost"},
                    delta={'reference': base_encounter_cost * 0.3, 'position': "top", 'valueformat': ",.2f"},
                    gauge={
                        'axis': {'range': [0, max_range], 'tickwidth': 1},
                        'bar': {'color': "#1E88E5"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#E3F2FD",
                        'steps': [
                            {'range': [0, max_range/3], 'color': '#E3F2FD'},
                            {'range': [max_range/3, max_range*2/3], 'color': '#90CAF9'},
                            {'range': [max_range*2/3, max_range], 'color': '#42A5F5'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': predicted_cost
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    height=300,
                    font={'color': "#424242", 'family': "Arial"},
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with viz_col2:
                # Cost Breakdown Pie Chart
                total_cost = base_encounter_cost + total_proc_base_cost
                cost_components = {
                    'Base Encounter': base_encounter_cost,
                    'Procedures': total_proc_base_cost,
                    'Estimated Out-of-Pocket': predicted_cost
                }
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(cost_components.keys()),
                    values=list(cost_components.values()),
                    hole=.4,
                    marker_colors=['#1E88E5', '#90CAF9', '#E3F2FD']
                )])
                
                fig_pie.update_layout(
                    title="Cost Breakdown",
                    height=300,
                    font={'color': "#424242", 'family': "Arial"},
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # Cost Breakdown and Insights
            st.markdown('<h2 class="subheader">Detailed Analysis</h2>', unsafe_allow_html=True)
            col6, col7, col8 = st.columns(3)
            
            with col6:
                st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Total Base Cost</p>
                        <p class="metric-value">${total_cost:,.2f}</p>
                        <p style="color: #4CAF50; font-size: 0.9rem;">Total before insurance</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col7:
                coverage_percentage = ((total_cost - predicted_cost) / total_cost * 100) if total_cost > 0 else 0
                st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Estimated Coverage</p>
                        <p class="metric-value">{coverage_percentage:.1f}%</p>
                        <p style="color: #4CAF50; font-size: 0.9rem;">Based on prediction</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col8:
                st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Your Responsibility</p>
                        <p class="metric-value">${predicted_cost:,.2f}</p>
                        <p style="color: #4CAF50; font-size: 0.9rem;">Estimated out-of-pocket</p>
                    </div>
                """, unsafe_allow_html=True)

            # Risk Factors and Recommendations
            st.markdown('<h2 class="subheader">Risk Factors & Recommendations</h2>', unsafe_allow_html=True)
            col9, col10 = st.columns(2)
            
            with col9:
                st.markdown(f"""
                    <div class="insight-card">
                        <h4 style="color: #0277BD; margin-bottom: 1rem;">Cost Impact Factors</h4>
                        <ul style="list-style-type: none; padding-left: 0;">
                            <li style="margin-bottom: 0.5rem;">🏥 <strong>Encounter Type:</strong> {encounterclass.capitalize()}</li>
                            <li style="margin-bottom: 0.5rem;">🔬 <strong>Procedures:</strong> {num_procedures} scheduled (${total_proc_base_cost:,.2f})</li>
                            <li style="margin-bottom: 0.5rem;">📋 <strong>Prior Conditions:</strong> {num_prior_conditions}</li>
                            <li style="margin-bottom: 0.5rem;">🏢 <strong>Insurance Provider:</strong> {payer_name}</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            
            with col10:
                cost_ratio = predicted_cost / total_cost if total_cost > 0 else 0
                recommendations = []
                
                if cost_ratio > 0.4:
                    recommendations.append("💡 Consider reviewing insurance coverage options")
                if num_procedures > 0:
                    recommendations.append("🏥 Review procedure necessity and timing")
                if total_cost > 5000:
                    recommendations.append("💳 Inquire about payment plan options")
                if num_prior_conditions > 3:
                    recommendations.append("📋 Schedule preventive care consultation")
                if payer_name == 'NO_INSURANCE':
                    recommendations.append("🏥 Consider applying for Medicaid or Medicare if eligible")
                    recommendations.append("💡 Research marketplace insurance options")
                
                recommendations_html = "\n".join([
                    f'<li style="margin-bottom: 0.5rem;">{rec}</li>'
                    for rec in recommendations
                ])
                
                st.markdown(f"""
                    <div class="insight-card">
                        <h4 style="color: #0277BD; margin-bottom: 1rem;">Recommendations</h4>
                        <ul style="list-style-type: none; padding-left: 0;">
                            {recommendations_html}
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"⚠️ Error making prediction: {str(e)}")
            st.error("Please check if all input values are in the expected format and range.")
            st.error(f"Input data shape: {input_data.shape if 'input_data' in locals() else 'unknown'}")
            st.error("Features used: " + ", ".join(FEATURES))

# Footer
st.markdown("""
    <div class="highlight-box" style="margin-top: 2rem;">
        <p style="color: #666666; text-align: center; margin-bottom: 0;">
            ⚠️ This is an estimation tool. Actual costs may vary based on specific insurance plans and provider policies.
            <br>Predictions are based on historical data and current market trends.
        </p>
    </div>
""", unsafe_allow_html=True) 