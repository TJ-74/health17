import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

#deploy this

#deploy again

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

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #2563eb;
        --secondary-color: #3b82f6;
        --accent-color: #60a5fa;
        --background-color: #f8fafc;
        --card-background: #ffffff;
        --text-color: #1e293b;
        --muted-text: #64748b;
        --border-color: #e2e8f0;
        --hover-color: #f1f5f9;
        --input-border: #cbd5e1;
        --input-focus: #2563eb;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
    }

    /* Global styles */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: var(--background-color);
    }

    /* Header styles */
    .custom-header {
        color: var(--text-color);
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--border-color);
        transition: all 0.3s ease;
    }

    .custom-header:hover {
        color: var(--primary-color);
    }

    .subheader {
        color: var(--text-color);
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0 1.5rem;
        transition: all 0.3s ease;
    }

    .subheader:hover {
        color: var(--primary-color);
    }

    /* Card styles */
    .metric-card {
        background-color: var(--card-background);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    /* Input styles */
    .stNumberInput>div>div>input,
    .stSelectbox>div>div,
    .stTextInput>div>div>input {
        background-color: var(--card-background) !important;
        border: 2px solid var(--input-border) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        color: var(--text-color) !important;
        font-size: 1rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div:focus,
    .stTextInput>div>div>input:focus {
        border-color: var(--input-focus) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2) !important;
        outline: none !important;
    }

    .stNumberInput>div>div>input:hover,
    .stSelectbox>div>div:hover,
    .stTextInput>div>div>input:hover {
        border-color: var(--input-focus) !important;
    }

    /* Button styles */
    .stButton>button {
        width: 100%;
        background-color: var(--primary-color);
        color: white;
        border-radius: 12px;
        height: 3.5em;
        margin-top: 2em;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.1), 0 2px 4px -1px rgba(37, 99, 235, 0.06);
    }

    .stButton>button:hover {
        background-color: var(--secondary-color);
        transform: translateY(-2px);
        box-shadow: 0 8px 12px -3px rgba(37, 99, 235, 0.2), 0 4px 6px -2px rgba(37, 99, 235, 0.1);
    }

    /* Cost display */
    .cost-display {
        background-color: var(--card-background);
        border-radius: 16px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        border: 2px solid var(--border-color);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .cost-display:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    .cost-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: var(--primary-color);
        margin: 1rem 0;
        text-shadow: 0 2px 4px rgba(37, 99, 235, 0.1);
    }

    .cost-label {
        font-size: 1.2rem;
        color: var(--muted-text);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    /* Insight card */
    .insight-card {
        background-color: var(--card-background);
        border-left: 4px solid var(--primary-color);
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        border-radius: 0 16px 16px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .insight-card:hover {
        transform: translateX(4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    /* Highlight box */
    .highlight-box {
        background-color: var(--card-background);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1.5rem 0;
        border: 2px solid var(--border-color);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .highlight-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    /* Loading animation */
    .stProgress > div > div > div > div {
        background-color: var(--primary-color);
    }

    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: var(--text-color);
        color: var(--card-background);
        text-align: center;
        padding: 8px 12px;
        border-radius: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
""", unsafe_allow_html=True)

# Define the features used in the model - Order must match exactly with the model's expected order
FEATURES = [
    'payer_name', 'age', 'race', 'income', 
    'encounterclass', 'base_encounter_cost', 'num_procedures', 
    'total_proc_base_cost', 'num_prior_conditions'
]
TARGET = 'out_of_pocket'

# Load the trained model
try:
    model = joblib.load('xgboost_tuned_model.pkl')
    # Convert to CPU if needed
    if hasattr(model, 'get_booster'):
        try:
            model.get_booster().set_param({'predictor': 'cpu_predictor'})
        except:
            pass  # Ignore if setting parameter fails
    st.sidebar.success('✅ Model loaded successfully')
except Exception as e:
    st.sidebar.error('❌ Error loading model: Make sure xgboost_tuned_model.pkl is available')
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

RACE_ENCODING = {
    'asian': 0,
    'black': 1,
    'hawaiian': 2,
    'native': 3,
    'other': 4,
    'white': 5
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

# Update the gauge chart colors and style
def create_gauge_chart(value, max_range, reference):
    return go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={
            'prefix': "$",
            'valueformat': ",.2f",
            'font': {'color': "#212121", 'size': 28, 'family': "Arial"}
        },
        title={
            'text': "Estimated Out-of-Pocket Cost",
            'font': {'color': "#1976D2", 'size': 20, 'family': "Arial"}
        },
        delta={
            'reference': reference,
            'position': "top",
            'valueformat': ",.2f",
            'font': {'color': "#757575"}
        },
        gauge={
            'axis': {
                'range': [0, max_range],
                'tickwidth': 1,
                'tickcolor': "#E0E0E0",
                'tickfont': {'color': "#757575"}
            },
            'bar': {'color': "#1976D2"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E0E0E0",
            'steps': [
                {'range': [0, max_range/3], 'color': "#E3F2FD"},
                {'range': [max_range/3, max_range*2/3], 'color': "#BBDEFB"},
                {'range': [max_range*2/3, max_range], 'color': "#90CAF9"}
            ],
            'threshold': {
                'line': {'color': "#F44336", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

# Update the pie chart colors and style
def create_pie_chart(cost_components):
    return go.Figure(data=[go.Pie(
        labels=list(cost_components.keys()),
        values=list(cost_components.values()),
        hole=.7,
        marker_colors=['#2563eb', '#3b82f6', '#60a5fa'],
        textfont={'color': "#1e293b", 'size': 14, 'family': "Inter"},
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<extra></extra>",
        textposition='outside',
        textinfo='label+percent',
        pull=[0.02, 0.02, 0.02]
    )])

# Add function to create insurance comparison bar chart
def create_insurance_comparison_chart(current_inputs, model, features):
    # Store predictions for each insurance provider
    predictions = []
    for provider, code in PAYER_ENCODING.items():
        # Copy current inputs and change only the insurance provider
        provider_inputs = current_inputs.copy()
        provider_inputs['payer_name'] = code
        
        # Create DataFrame with exact feature order
        input_df = pd.DataFrame([provider_inputs])[features]
        
        # Make prediction
        pred_cost = model.predict(input_df)[0]
        pred_cost = max(0, pred_cost)  # Ensure non-negative
        
        predictions.append({
            'Insurance Provider': provider,
            'Estimated Cost': pred_cost
        })
    
    # Create DataFrame for plotting
    df_predictions = pd.DataFrame(predictions)
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=df_predictions['Insurance Provider'],
            y=df_predictions['Estimated Cost'],
            marker_color='#60a5fa',
            hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
            text=df_predictions['Estimated Cost'].apply(lambda x: f'${x:,.2f}'),
            textposition='auto',
            textfont={'family': "Inter"},
            marker=dict(
                line=dict(width=1, color='#e2e8f0')
            )
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Cost Comparison Across Insurance Providers",
            'font': {'color': "#1e293b", 'size': 20, 'family': "Inter"},
            'y': 0.95
        },
        xaxis={
            'title': None,
            'tickangle': -45,
            'tickfont': {'color': "#1e293b", 'family': "Inter"},
            'gridcolor': "#f1f5f9",
            'showgrid': False
        },
        yaxis={
            'title': "Estimated Out-of-Pocket Cost ($)",
            'tickformat': "$,.0f",
            'tickfont': {'color': "#1e293b", 'family': "Inter"},
            'gridcolor': "#f1f5f9",
            'ticksuffix': " "
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=450,
        showlegend=False,
        margin=dict(l=60, r=20, t=80, b=120),
        font={'color': "#1e293b", 'family': "Inter"}
    )
    
    # Add current provider marker
    current_pred = df_predictions[df_predictions['Insurance Provider'] == payer_name]['Estimated Cost'].values[0]
    fig.add_trace(go.Scatter(
        x=[payer_name],
        y=[current_pred],
        mode='markers',
        marker=dict(
            symbol='star',
            size=20,
            color='#FF6B6B',
        ),
        name='Current Selection',
        hovertemplate="<b>Current Selection</b><br>$%{y:,.2f}<extra></extra>"
    ))
    
    return fig

# Add function to create income vs cost line graph
def create_income_analysis_chart(current_inputs, model, features):
    # Generate income range from 5000 to 50000 with 5k steps
    income_range = list(range(5000, 55000, 5000))
    predictions = []
    
    # Make predictions for each income level
    for income in income_range:
        # Copy current inputs and change only the income
        income_inputs = current_inputs.copy()
        income_inputs['income'] = income
        
        # Create DataFrame with exact feature order
        input_df = pd.DataFrame([income_inputs])[features]
        
        # Make prediction
        pred_cost = model.predict(input_df)[0]
        pred_cost = max(0, pred_cost)  # Ensure non-negative
        
        predictions.append({
            'Annual Income': income,
            'Estimated Cost': pred_cost
        })
    
    # Create DataFrame for plotting
    df_predictions = pd.DataFrame(predictions)
    
    # Create line chart
    fig = go.Figure()
    
    # Add line
    fig.add_trace(go.Scatter(
        x=df_predictions['Annual Income'],
        y=df_predictions['Estimated Cost'],
        mode='lines+markers',
        name='Estimated Cost',
        line=dict(color='#2563eb', width=3, shape='spline'),
        marker=dict(size=8, symbol='circle', color='#3b82f6'),
        hovertemplate="<b>Income: $%{x:,.0f}</b><br>Cost: $%{y:,.2f}<extra></extra>"
    ))
    
    # Add current income point
    fig.add_trace(go.Scatter(
        x=[current_inputs['income']],
        y=[df_predictions[df_predictions['Annual Income'] == 
           (current_inputs['income'] - current_inputs['income'] % 5000)]['Estimated Cost'].values[0]],
        mode='markers',
        name='Current Income',
        marker=dict(
            symbol='star',
            size=20,
            color='#FF6B6B',
        ),
        hovertemplate="<b>Your Income: $%{x:,.0f}</b><br>Cost: $%{y:,.2f}<extra></extra>"
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Income vs. Out-of-Pocket Cost Analysis",
            'font': {'color': "#1e293b", 'size': 20, 'family': "Inter"},
            'y': 0.95
        },
        xaxis={
            'title': "Annual Income ($)",
            'tickformat': "$,.0f",
            'tickfont': {'color': "#1e293b", 'family': "Inter"},
            'gridcolor': "#f1f5f9",
            'ticksuffix': " "
        },
        yaxis={
            'title': "Estimated Out-of-Pocket Cost ($)",
            'tickformat': "$,.0f",
            'tickfont': {'color': "#1e293b", 'family': "Inter"},
            'gridcolor': "#f1f5f9",
            'ticksuffix': " "
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=450,
        showlegend=True,
        legend={
            'font': {'color': "#1e293b", 'family': "Inter"},
            'bgcolor': "#f8fafc",
            'bordercolor': "#e2e8f0"
        }
    )
    
    return fig

# Add function to create age vs cost line graph
def create_age_analysis_chart(current_inputs, model, features):
    # Generate age range from 1 to 70 with 4-year steps
    age_range = list(range(1, 71, 4))
    predictions = []
    
    try:
        # Make predictions for each age
        for age in age_range:
            # Copy current inputs and change only the age
            age_inputs = current_inputs.copy()
            age_inputs['age'] = age
            
            # Create DataFrame with exact feature order
            input_df = pd.DataFrame([age_inputs])[features]
            
            # Make prediction
            pred_cost = model.predict(input_df)[0]
            pred_cost = max(0, pred_cost)  # Ensure non-negative
            
            predictions.append({
                'Age': age,
                'Estimated Cost': pred_cost
            })
        
        # Create DataFrame for plotting
        df_predictions = pd.DataFrame(predictions)
        
        # Create line chart
        fig = go.Figure()
        
        # Add line
        fig.add_trace(go.Scatter(
            x=df_predictions['Age'],
            y=df_predictions['Estimated Cost'],
            mode='lines+markers',
            name='Estimated Cost',
            line=dict(color='#2563eb', width=3, shape='spline'),
            marker=dict(size=8, symbol='circle', color='#3b82f6'),
            hovertemplate="<b>Age: %{x} years</b><br>Cost: $%{y:,.2f}<extra></extra>"
        ))
        
        # Add current age point - with improved handling
        current_age = current_inputs['age']
        
        # Make a specific prediction for the current age
        current_age_inputs = current_inputs.copy()
        current_age_inputs['age'] = current_age
        current_age_df = pd.DataFrame([current_age_inputs])[features]
        current_age_prediction = max(0, model.predict(current_age_df)[0])
        
        fig.add_trace(go.Scatter(
            x=[current_age],
            y=[current_age_prediction],
            mode='markers',
            name='Current Age',
            marker=dict(
                symbol='star',
                size=20,
                color='#FF6B6B',
            ),
            hovertemplate="<b>Your Age: %{x} years</b><br>Cost: $%{y:,.2f}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Age vs. Out-of-Pocket Cost Analysis",
                'font': {'color': "#1e293b", 'size': 20, 'family': "Inter"},
                'y': 0.95
            },
            xaxis={
                'title': "Age (years)",
                'tickfont': {'color': "#1e293b", 'family': "Inter"},
                'gridcolor': "#f1f5f9",
                'dtick': 5
            },
            yaxis={
                'title': "Estimated Out-of-Pocket Cost ($)",
                'tickformat': "$,.0f",
                'tickfont': {'color': "#1e293b", 'family': "Inter"},
                'gridcolor': "#f1f5f9",
                'ticksuffix': " "
            },
            paper_bgcolor="white",
            plot_bgcolor="white",
            height=450,
            showlegend=True,
            legend={
                'font': {'color': "#1e293b", 'family': "Inter"},
                'bgcolor': "#f8fafc",
                'bordercolor': "#e2e8f0"
            }
        )
        
        return fig
    except Exception as e:
        st.error(f"Error in age analysis: {str(e)}")
        st.error("Unable to generate age analysis chart. Please check your inputs.")
        return None

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
    race = st.selectbox("Race", list(RACE_ENCODING.keys()), format_func=lambda x: x.capitalize(), key="race_select")
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
        st.error("⚠️ Model not loaded. Please ensure xgboost_tuned_model.pkl is available.")
    else:
        try:
            # Create input data dictionary with all features
            input_dict = {
                'payer_name': PAYER_ENCODING[payer_name],
                'age': age,
                'race': RACE_ENCODING[race],
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
                st.markdown(f"""
                    <div class="cost-display">
                        <p class="cost-label">Estimated Out-of-Pocket Cost</p>
                        <p class="cost-value">${predicted_cost:,.2f}</p>
                    </div>
                """, unsafe_allow_html=True)

            with viz_col2:
                total_cost = base_encounter_cost + total_proc_base_cost
                cost_components = {
                    'Base Encounter': base_encounter_cost,
                    'Procedures': total_proc_base_cost,
                    'Estimated Out-of-Pocket': predicted_cost
                }
                
                fig_pie = create_pie_chart(cost_components)
                fig_pie.update_layout(
                    title={
                        'text': "Cost Breakdown",
                        'font': {'color': "#1976D2", 'size': 20},
                        'y': 0.95
                    },
                    height=350,
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    showlegend=True,
                    legend={
                        'font': {'color': "#212121"},
                        'bgcolor': "#F5F5F5",
                        'bordercolor': "#E0E0E0"
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # Add Insurance Comparison Section
            st.markdown('<h2 class="subheader">Insurance Provider Comparison</h2>', unsafe_allow_html=True)
            
            # Create comparison chart
            fig_comparison = create_insurance_comparison_chart(input_dict, model, FEATURES)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Add explanation box
            st.markdown("""
                <div class="highlight-box">
                    <p style="color: #212121; margin-bottom: 0.5rem;">
                        <strong>💡 Cost Comparison Insights:</strong>
                    </p>
                    <p style="color: #757575; font-size: 0.9rem; margin-bottom: 0;">
                        This chart shows estimated out-of-pocket costs across all insurance providers for your specific case.
                        The red star (⭐) indicates your current selection. Compare different providers to find potential cost savings.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Add Income Analysis Section
            st.markdown('<h2 class="subheader">Income Impact Analysis</h2>', unsafe_allow_html=True)
            
            # Create income analysis chart
            fig_income = create_income_analysis_chart(input_dict, model, FEATURES)
            st.plotly_chart(fig_income, use_container_width=True)
            
            # Add explanation box
            st.markdown("""
                <div class="highlight-box">
                    <p style="color: #212121; margin-bottom: 0.5rem;">
                        <strong>💡 Income Impact Insights:</strong>
                    </p>
                    <p style="color: #757575; font-size: 0.9rem; margin-bottom: 0;">
                        This graph shows how estimated out-of-pocket costs vary with different income levels, 
                        keeping all other factors constant. The red star (⭐) indicates your current income level.
                        Understanding this relationship can help in financial planning and choosing appropriate insurance coverage.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Add Age Analysis Section
            st.markdown('<h2 class="subheader">Age Impact Analysis</h2>', unsafe_allow_html=True)
            
            # Create age analysis chart with error handling
            fig_age = create_age_analysis_chart(input_dict, model, FEATURES)
            if fig_age is not None:
                st.plotly_chart(fig_age, use_container_width=True)
                
                # Add explanation box
                st.markdown("""
                    <div class="highlight-box">
                        <p style="color: #212121; margin-bottom: 0.5rem;">
                            <strong>💡 Age Impact Insights:</strong>
                        </p>
                        <p style="color: #757575; font-size: 0.9rem; margin-bottom: 0;">
                            This graph illustrates how estimated out-of-pocket costs vary across different age groups, 
                            keeping all other factors constant. The red star (⭐) marks your current age.
                            Understanding this relationship can help in long-term healthcare financial planning and 
                            anticipating potential cost changes as you age.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

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
                # Calculate coverage amount and percentage
                if total_cost > 0:
                    covered_amount = max(0, total_cost - predicted_cost)  # Ensure covered amount is not negative
                    coverage_percentage = min(100, max(0, (covered_amount / total_cost) * 100))  # Clamp between 0-100%
                else:
                    coverage_percentage = 0
                
                # Determine coverage level for color coding
                if coverage_percentage >= 80:
                    coverage_color = "#4CAF50"  # Green for high coverage
                elif coverage_percentage >= 50:
                    coverage_color = "#FFA726"  # Orange for medium coverage
                else:
                    coverage_color = "#EF5350"  # Red for low coverage
                
                st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Insurance Coverage</p>
                        <p class="metric-value" style="color: {coverage_color};">{coverage_percentage:.1f}%</p>
                        <p style="color: {coverage_color}; font-size: 0.9rem;">
                            ${covered_amount:,.2f} covered by insurance
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col8:
                responsibility_percentage = min(100, max(0, (predicted_cost / total_cost * 100))) if total_cost > 0 else 0
                st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Your Responsibility</p>
                        <p class="metric-value">${predicted_cost:,.2f}</p>
                        <p style="color: var(--muted-text); font-size: 0.9rem;">
                            {responsibility_percentage:.1f}% of total cost
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            # Add a coverage explanation box if needed
            if coverage_percentage < 20 and payer_name != 'NO_INSURANCE':
                st.markdown("""
                    <div class="highlight-box" style="border-left: 4px solid #EF5350;">
                        <p style="color: #EF5350; margin-bottom: 0.5rem;">
                            <strong>⚠️ Low Coverage Alert:</strong>
                        </p>
                        <p style="color: #757575; font-size: 0.9rem; margin-bottom: 0;">
                            The estimated coverage is unusually low. This might be due to:
                            <ul style="margin-top: 0.5rem;">
                                <li>Services not covered under the selected plan</li>
                                <li>High deductible not yet met</li>
                                <li>Out-of-network providers</li>
                            </ul>
                            Consider consulting with your insurance provider for detailed coverage information.
                        </p>
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
