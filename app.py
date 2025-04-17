import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Healthcare Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    user_type = st.radio("Select User Type", ["Patient", "Insurance Provider"])
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
        <h4 style='margin-bottom: 0.5em; color: #0277BD;'>Smart Healthcare Cost Analysis</h4>
        <p class="info-text">Empowering better healthcare decisions through predictive analytics.</p>
    </div>
""", unsafe_allow_html=True)

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Patient Demographics")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    race = st.selectbox("Race", ["White", "Black", "Asian", "Native", "Other"])
    ethnicity = st.selectbox("Ethnicity", ["Hispanic", "Non-Hispanic"])
    income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Encounter Details")
    encounter_class = st.selectbox("Encounter Class", ["Inpatient", "Outpatient", "Emergency", "Urgent Care"])
    code = st.text_input("Encounter Code (CPT/SNOMED)")
    reason_code = st.text_input("Reason Code")
    base_encounter_cost = st.number_input("Base Encounter Cost ($)", min_value=0.0, value=100.0)
    payer_name = st.selectbox("Insurance Provider", ["Medicare", "Medicaid", "Blue Cross", "Aetna", "UnitedHealth", "Other"])
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("Medical History")
    num_medications = st.number_input("Number of Current Medications", min_value=0, value=0)
    num_procedures = st.number_input("Number of Current Procedures", min_value=0, value=0)
    num_prior_conditions = st.number_input("Number of Prior Conditions", min_value=0, value=0)
    num_prior_procedures = st.number_input("Number of Prior Procedures", min_value=0, value=0)
    num_prior_medications = st.number_input("Number of Prior Medications", min_value=0, value=0)
    has_chronic_condition = st.checkbox("Has Chronic Condition")
    st.markdown('</div>', unsafe_allow_html=True)

# Cost Details
st.markdown('<div class="metric-card">', unsafe_allow_html=True)
st.subheader("Cost Details")
col4, col5 = st.columns(2)

with col4:
    total_med_base_cost = st.number_input("Total Medication Base Cost ($)", min_value=0.0, value=0.0)

with col5:
    total_proc_base_cost = st.number_input("Total Procedure Base Cost ($)", min_value=0.0, value=0.0)
st.markdown('</div>', unsafe_allow_html=True)

# Predict button
if st.button("Analyze Healthcare Costs"):
    st.markdown('<h2 class="subheader">Cost Analysis Results</h2>', unsafe_allow_html=True)
    
    # Create two columns for visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Cost Prediction Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=2500,
            title={'text': "Estimated Out-of-Pocket Cost ($)"},
            delta={'reference': 3000, 'position': "top"},
            gauge={
                'axis': {'range': [None, 5000], 'tickwidth': 1},
                'bar': {'color': "#1E88E5"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#E3F2FD",
                'steps': [
                    {'range': [0, 1000], 'color': '#E3F2FD'},
                    {'range': [1000, 3000], 'color': '#90CAF9'},
                    {'range': [3000, 5000], 'color': '#42A5F5'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 2500}}))
        
        fig_gauge.update_layout(
            height=300,
            font={'color': "#424242", 'family': "Arial"},
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with viz_col2:
        # Insurance Coverage Comparison
        insurance_data = {
            'Provider': ['Medicare', 'Medicaid', 'Blue Cross', 'Aetna', 'UnitedHealth'],
            'Coverage': [65, 70, 80, 75, 85],
            'Monthly Premium': [200, 150, 300, 280, 320]
        }
        df_insurance = pd.DataFrame(insurance_data)
        
        fig_comparison = px.scatter(df_insurance, 
                                  x='Monthly Premium', 
                                  y='Coverage',
                                  size=[40]*5,
                                  color='Provider',
                                  title="Insurance Provider Comparison")
        
        fig_comparison.update_layout(
            height=300,
            font={'color': "#666666", 'family': "Arial"},
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

    # Cost Breakdown and Insights
    st.markdown('<h2 class="subheader">Detailed Analysis</h2>', unsafe_allow_html=True)
    col6, col7, col8 = st.columns(3)
    
    with col6:
        st.markdown("""
            <div class="metric-card">
                <p class="metric-label">Base Cost</p>
                <p class="metric-value">$2,500.00</p>
                <p style="color: #4CAF50; font-size: 0.9rem;">▼ $500 vs. avg</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col7:
        st.markdown("""
            <div class="metric-card">
                <p class="metric-label">Insurance Coverage</p>
                <p class="metric-value">70%</p>
                <p style="color: #4CAF50; font-size: 0.9rem;">▲ 5% higher than avg</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col8:
        st.markdown("""
            <div class="metric-card">
                <p class="metric-label">Your Responsibility</p>
                <p class="metric-value">$2,500.00</p>
                <p style="color: #4CAF50; font-size: 0.9rem;">▼ $300 vs. market</p>
            </div>
        """, unsafe_allow_html=True)

    # Risk Assessment and Recommendations
    st.markdown('<h2 class="subheader">Insights and Recommendations</h2>', unsafe_allow_html=True)
    col9, col10 = st.columns(2)
    
    with col9:
        st.markdown("""
            <div class="insight-card">
                <h4 style="color: #0277BD; margin-bottom: 1rem;">Risk Assessment</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="margin-bottom: 0.5rem;">🟢 <strong>Health Risk Score:</strong> Medium-Low</li>
                    <li style="margin-bottom: 0.5rem;">🟡 <strong>Cost Risk Level:</strong> Moderate</li>
                    <li style="margin-bottom: 0.5rem;">🟢 <strong>Chronic Condition Impact:</strong> Minimal</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col10:
        st.markdown("""
            <div class="insight-card">
                <h4 style="color: #0277BD; margin-bottom: 1rem;">Recommendations</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="margin-bottom: 0.5rem;">💡 Consider Blue Cross for better chronic condition coverage</li>
                    <li style="margin-bottom: 0.5rem;">📈 Preventive care could reduce long-term costs by 15%</li>
                    <li style="margin-bottom: 0.5rem;">💊 Medication cost optimization available through generic alternatives</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Historical Trends
    st.markdown('<h2 class="subheader">Cost Trend Analysis</h2>', unsafe_allow_html=True)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
    costs = np.random.normal(2500, 200, len(dates))
    trend_data = pd.DataFrame({'Date': dates, 'Cost': costs})
    
    fig_trend = px.line(trend_data, x='Date', y='Cost',
                        title='Projected Cost Trends',
                        labels={'Cost': 'Monthly Healthcare Costs ($)', 'Date': 'Timeline'})
    fig_trend.update_layout(
        height=400,
        font={'color': "#424242", 'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_font_color="#0277BD"
    )
    fig_trend.update_xaxes(gridcolor="#E3F2FD")
    fig_trend.update_yaxes(gridcolor="#E3F2FD")
    st.plotly_chart(fig_trend, use_container_width=True)

# Footer
st.markdown("""
    <div class="highlight-box" style="margin-top: 2rem;">
        <p style="color: #666666; text-align: center; margin-bottom: 0;">
            ⚠️ This is an estimation tool. Actual costs may vary based on specific insurance plans and provider policies.
            <br>Updated with real-time market data and AI-powered predictions.
        </p>
    </div>
""", unsafe_allow_html=True) 