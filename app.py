import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
# ─────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ford DriveValue · Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─────────────────────────────────────────────────────────────────
# Custom CSS for Premium UI
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -30%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e0e7ff, #c7d2fe, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1b4b, #312e81);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99,102,241,0.2);
        border-color: rgba(99,102,241,0.4);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .section-header-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(99,102,241,0.5), transparent);
    }
    /* Prediction Result */
    .prediction-box {
        background: linear-gradient(145deg, #064e3b, #065f46);
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(16,185,129,0.15);
        margin: 1rem 0;
    }
    .prediction-price {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #6ee7b7, #34d399, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .prediction-label {
        font-size: 0.85rem;
        color: #6ee7b7;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 600;
    }
    .prediction-subtitle {
        color: #a7f3d0;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1e1b4b) !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #c7d2fe !important;
        font-weight: 500 !important;
    }
    /* Info Box */
    .info-box {
        background: linear-gradient(145deg, #1e1b4b, #312e81);
        border-left: 4px solid #6366f1;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #c7d2fe;
        font-size: 0.9rem;
    }
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.8rem;
        border-top: 1px solid rgba(99,102,241,0.1);
        margin-top: 3rem;
    }
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────
# Load Data & Train Model (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(dir_path, "ford.csv")
    if not os.path.exists(csv_path):
        csv_path = "Ford_Drive_Value/ford.csv"
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['model'] = df['model'].str.strip()
    df['transmission'] = df['transmission'].str.strip()
    df['fuelType'] = df['fuelType'].str.strip()
    return df
@st.cache_resource
def train_model(df):
    """Train LinearRegression with one-hot encoding (matching notebook approach)."""
    y = df['price']
    X = df.drop(columns=['price'])
    # One-hot encode categorical columns
    categorical_cols = ['model', 'transmission', 'fuelType']
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    # Standardize numerical features
    numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.33, random_state=42
    )
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Predictions & metrics
    y_pred = model.predict(X_test)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mse': mean_squared_error(y_test, y_pred),
    }
    return model, scaler, X_encoded.columns.tolist(), metrics, X_test, y_test, y_pred
# Load & process
df = load_data()
model, scaler, feature_cols, metrics, X_test, y_test, y_pred = train_model(df)
# ─────────────────────────────────────────────────────────────────
# Hero Section
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🔬 MACHINE LEARNING · LINEAR REGRESSION</div>
    <div class="hero-title">Ford DriveValue</div>
    <div class="hero-subtitle">
        Predict used Ford car prices with 83% accuracy using a One-Hot Encoded Linear Regression model
        trained on 17,966 real-world listings.
    </div>
</div>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────
# Model Performance Metrics
# ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{metrics['r2']:.1%}</div>
        <div class="metric-label">R² Score (Accuracy)</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">£{metrics['mae']:,.0f}</div>
        <div class="metric-label">Mean Absolute Error</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">£{metrics['rmse']:,.0f}</div>
        <div class="metric-label">Root Mean Sq Error</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Training Samples</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────
# Sidebar – Prediction Inputs
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0;">
        <div style="font-size:2.5rem;">🚗</div>
        <div style="font-size:1.2rem; font-weight:700; color:#c7d2fe; margin-top:0.5rem;">
            Price Predictor
        </div>
        <div style="font-size:0.8rem; color:#818cf8;">Configure your Ford below</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    # Model selection
    models_list = sorted(df['model'].unique().tolist())
    selected_model = st.selectbox("🚘 Model", models_list, index=models_list.index('Fiesta') if 'Fiesta' in models_list else 0)
    # Year
    selected_year = st.slider(
        "📅 Year",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=2018
    )
    # Transmission
    transmissions = sorted(df['transmission'].unique().tolist())
    selected_transmission = st.selectbox("⚙️ Transmission", transmissions)
    # Fuel Type
    fuel_types = sorted(df['fuelType'].unique().tolist())
    selected_fuel = st.selectbox("⛽ Fuel Type", fuel_types)
    # Mileage
    selected_mileage = st.slider(
        "🛣️ Mileage",
        min_value=0,
        max_value=int(df['mileage'].max()),
        value=20000,
        step=500
    )
    # Engine Size
    engine_sizes = sorted(df['engineSize'].unique().tolist())
    selected_engine = st.select_slider("🔧 Engine Size (L)", options=engine_sizes, value=1.0)
    # Tax
    selected_tax = st.slider(
        "💰 Tax (£)",
        min_value=int(df['tax'].min()),
        max_value=int(df['tax'].max()),
        value=145
    )
    # MPG
    selected_mpg = st.slider(
        "⛽ MPG",
        min_value=float(df['mpg'].min()),
        max_value=float(df['mpg'].max()),
        value=55.0,
        step=0.5
    )
    st.markdown("---")
    # Predict button
    predict_btn = st.button("🎯 Predict Price", use_container_width=True, type="primary")
# ─────────────────────────────────────────────────────────────────
# Make Prediction
# ─────────────────────────────────────────────────────────────────
def predict_price(model_name, year, transmission, fuel_type, mileage, engine_size, tax, mpg):
    """Create a one-hot encoded input and predict."""
    # Build raw input
    input_data = pd.DataFrame([{
        'year': year,
        'mileage': mileage,
        'tax': tax,
        'mpg': mpg,
        'engineSize': engine_size,
        'model': model_name,
        'transmission': transmission,
        'fuelType': fuel_type,
    }])
    # One-hot encode
    categorical_cols = ['model', 'transmission', 'fuelType']
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)
    # Standardize numerical features
    numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])
    # Align columns with training data
    for col in feature_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_cols]
    prediction = model.predict(input_encoded)[0]
    return max(prediction, 0)  # No negative prices
# Always calculate a prediction for display
predicted_price = predict_price(
    selected_model, selected_year, selected_transmission,
    selected_fuel, selected_mileage, selected_engine,
    selected_tax, selected_mpg
)
# ─────────────────────────────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Prediction",
    "📊 Data Explorer",
    "🔬 Model Insights",
    "📋 Raw Data"
])
# ═══════════════════════════════════════════════════════════════
# TAB 1: PREDICTION
# ═══════════════════════════════════════════════════════════════
with tab1:
    pred_col1, pred_col2 = st.columns([2, 3])
    with pred_col1:
        st.markdown("""
        <div class="section-header">
            🎯 Estimated Price
            <span class="section-header-line"></span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">Estimated Market Value</div>
            <div class="prediction-price">£{predicted_price:,.0f}</div>
            <div class="prediction-subtitle">
                {selected_year} Ford {selected_model} · {selected_transmission} · {selected_fuel}
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Confidence info
        st.markdown(f"""
        <div class="info-box">
            <strong>📊 Model Confidence</strong><br>
            R² Score: <strong>{metrics['r2']:.1%}</strong> · Avg Error: <strong>±£{metrics['mae']:,.0f}</strong><br>
            <span style="font-size:0.8rem; color:#94a3b8;">
                Prediction range: £{max(predicted_price - metrics['mae'], 0):,.0f} – £{predicted_price + metrics['mae']:,.0f}
            </span>
        </div>
        """, unsafe_allow_html=True)
    with pred_col2:
        # Similar cars comparison
        st.markdown("""
        <div class="section-header">
            🔍 Similar Cars in Dataset
            <span class="section-header-line"></span>
        </div>
        """, unsafe_allow_html=True)
        similar = df[
            (df['model'] == selected_model) &
            (df['year'] == selected_year) &
            (df['transmission'] == selected_transmission)
        ]
        if len(similar) > 0:
            fig_similar = go.Figure()
            fig_similar.add_trace(go.Histogram(
                x=similar['price'],
                nbinsx=20,
                marker_color='rgba(99,102,241,0.6)',
                marker_line_color='rgba(129,140,248,0.8)',
                marker_line_width=1,
                name='Similar Cars'
            ))
            fig_similar.add_vline(
                x=predicted_price,
                line_dash="dash",
                line_color="#10b981",
                line_width=3,
                annotation_text=f"Prediction: £{predicted_price:,.0f}",
                annotation_font_color="#10b981",
                annotation_font_size=13,
            )
            fig_similar.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                margin=dict(l=20, r=20, t=40, b=40),
                title=dict(
                    text=f"Price Distribution: {selected_year} Ford {selected_model} ({selected_transmission})",
                    font=dict(size=14, color='#c7d2fe')
                ),
                xaxis_title="Price (£)",
                yaxis_title="Count",
                xaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
                yaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
            )
            st.plotly_chart(fig_similar, use_container_width=True)
            # Stats
            scol1, scol2, scol3 = st.columns(3)
            scol1.metric("📉 Min Price", f"£{similar['price'].min():,}")
            scol2.metric("📊 Avg Price", f"£{similar['price'].mean():,.0f}")
            scol3.metric("📈 Max Price", f"£{similar['price'].max():,}")
        else:
            st.info(f"No exact matches found for {selected_year} Ford {selected_model} ({selected_transmission}). Try adjusting filters.")
# ═══════════════════════════════════════════════════════════════
# TAB 2: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-header">
        📊 Interactive Data Explorer
        <span class="section-header-line"></span>
    </div>
    """, unsafe_allow_html=True)
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        # Price by Model
        model_avg = df.groupby('model')['price'].agg(['mean', 'count']).reset_index()
        model_avg.columns = ['Model', 'Average Price', 'Count']
        model_avg = model_avg.sort_values('Average Price', ascending=True)
        fig_model = go.Figure()
        fig_model.add_trace(go.Bar(
            y=model_avg['Model'],
            x=model_avg['Average Price'],
            orientation='h',
            marker=dict(
                color=model_avg['Average Price'],
                colorscale=[[0, '#6366f1'], [0.5, '#8b5cf6'], [1, '#c084fc']],
                line=dict(width=0),
            ),
            text=[f"£{p:,.0f}" for p in model_avg['Average Price']],
            textposition='outside',
            textfont=dict(color='#c7d2fe', size=11),
            hovertemplate="<b>%{y}</b><br>Avg Price: £%{x:,.0f}<extra></extra>",
        ))
        fig_model.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            margin=dict(l=20, r=80, t=40, b=20),
            title=dict(text="Average Price by Model", font=dict(size=15, color='#c7d2fe')),
            xaxis=dict(title="Average Price (£)", gridcolor='rgba(99,102,241,0.1)'),
            yaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
        )
        st.plotly_chart(fig_model, use_container_width=True)
    with ex_col2:
        # Price vs Mileage scatter
        sample_df = df.sample(min(2000, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample_df,
            x='mileage',
            y='price',
            color='fuelType',
            size='engineSize',
            hover_data=['model', 'year', 'transmission'],
            color_discrete_sequence=['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'],
            opacity=0.7,
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            title=dict(text="Price vs Mileage (by Fuel Type)", font=dict(size=15, color='#c7d2fe')),
            xaxis=dict(title="Mileage", gridcolor='rgba(99,102,241,0.1)'),
            yaxis=dict(title="Price (£)", gridcolor='rgba(99,102,241,0.1)'),
            legend=dict(
                bgcolor='rgba(0,0,0,0.3)',
                font=dict(color='#c7d2fe'),
            ),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    # Row 2
    ex_col3, ex_col4 = st.columns(2)
    with ex_col3:
        # Price by Year trend
        year_avg = df.groupby('year')['price'].agg(['mean', 'count']).reset_index()
        year_avg.columns = ['Year', 'Average Price', 'Count']
        fig_year = go.Figure()
        fig_year.add_trace(go.Scatter(
            x=year_avg['Year'],
            y=year_avg['Average Price'],
            mode='lines+markers',
            line=dict(color='#8b5cf6', width=3),
            marker=dict(size=10, color='#c084fc', line=dict(width=2, color='#8b5cf6')),
            fill='tozeroy',
            fillcolor='rgba(139,92,246,0.1)',
            hovertemplate="<b>%{x}</b><br>Avg: £%{y:,.0f}<extra></extra>",
        ))
        fig_year.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            title=dict(text="Average Price Trend by Year", font=dict(size=15, color='#c7d2fe')),
            xaxis=dict(title="Year", gridcolor='rgba(99,102,241,0.1)', dtick=1),
            yaxis=dict(title="Average Price (£)", gridcolor='rgba(99,102,241,0.1)'),
        )
        st.plotly_chart(fig_year, use_container_width=True)
    with ex_col4:
        # Transmission distribution with price
        trans_data = df.groupby('transmission')['price'].agg(['mean', 'median', 'count']).reset_index()
        trans_data.columns = ['Transmission', 'Mean', 'Median', 'Count']
        fig_trans = go.Figure()
        fig_trans.add_trace(go.Bar(
            x=trans_data['Transmission'],
            y=trans_data['Mean'],
            name='Mean Price',
            marker_color='rgba(99,102,241,0.7)',
            text=[f"£{p:,.0f}" for p in trans_data['Mean']],
            textposition='outside',
            textfont=dict(color='#c7d2fe'),
        ))
        fig_trans.add_trace(go.Bar(
            x=trans_data['Transmission'],
            y=trans_data['Median'],
            name='Median Price',
            marker_color='rgba(139,92,246,0.7)',
            text=[f"£{p:,.0f}" for p in trans_data['Median']],
            textposition='outside',
            textfont=dict(color='#c7d2fe'),
        ))
        fig_trans.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            title=dict(text="Price by Transmission Type", font=dict(size=15, color='#c7d2fe')),
            xaxis=dict(title="", gridcolor='rgba(99,102,241,0.1)'),
            yaxis=dict(title="Price (£)", gridcolor='rgba(99,102,241,0.1)'),
            barmode='group',
            legend=dict(bgcolor='rgba(0,0,0,0.3)', font=dict(color='#c7d2fe')),
        )
        st.plotly_chart(fig_trans, use_container_width=True)
    # Row 3 – Fuel type pie & Engine Size box
    ex_col5, ex_col6 = st.columns(2)
    with ex_col5:
        fuel_counts = df['fuelType'].value_counts().reset_index()
        fuel_counts.columns = ['Fuel Type', 'Count']
        fig_pie = go.Figure(go.Pie(
            labels=fuel_counts['Fuel Type'],
            values=fuel_counts['Count'],
            hole=0.55,
            marker_colors=['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'],
            textinfo='label+percent',
            textfont=dict(color='white', size=12),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            title=dict(text="Fuel Type Distribution", font=dict(size=15, color='#c7d2fe')),
            legend=dict(bgcolor='rgba(0,0,0,0.3)', font=dict(color='#c7d2fe')),
            annotations=[dict(
                text=f"<b>{len(df):,}</b><br>cars",
                x=0.5, y=0.5,
                font_size=16,
                font_color='#c7d2fe',
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with ex_col6:
        fig_box = px.box(
            df,
            x='engineSize',
            y='price',
            color='engineSize',
            color_discrete_sequence=['#6366f1', '#7c3aed', '#8b5cf6', '#a78bfa', '#c084fc', '#d8b4fe', '#ede9fe'],
        )
        fig_box.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            title=dict(text="Price Distribution by Engine Size", font=dict(size=15, color='#c7d2fe')),
            xaxis=dict(title="Engine Size (L)", gridcolor='rgba(99,102,241,0.1)'),
            yaxis=dict(title="Price (£)", gridcolor='rgba(99,102,241,0.1)'),
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)
# ═══════════════════════════════════════════════════════════════
# TAB 3: MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-header">
        🔬 Model Performance Analysis
        <span class="section-header-line"></span>
    </div>
    """, unsafe_allow_html=True)
    ins_col1, ins_col2 = st.columns(2)
    with ins_col1:
        # Actual vs Predicted
        sample_idx = np.random.RandomState(42).choice(len(y_test), size=min(500, len(y_test)), replace=False)
        y_test_arr = np.array(y_test)
        y_pred_arr = np.array(y_pred)
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=y_test_arr[sample_idx],
            y=y_pred_arr[sample_idx],
            mode='markers',
            marker=dict(
                size=5,
                color=np.abs(y_test_arr[sample_idx] - y_pred_arr[sample_idx]),
                colorscale='Viridis',
                colorbar=dict(title=dict(text="Error (£)", font=dict(color='#c7d2fe')), tickfont=dict(color='#c7d2fe')),
                opacity=0.6,
                line=dict(width=0),
            ),
            hovertemplate="Actual: £%{x:,.0f}<br>Predicted: £%{y:,.0f}<extra></extra>",
        ))
        # Perfect prediction line
        max_val = max(y_test_arr.max(), y_pred_arr.max())
        fig_avp.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(dash='dash', color='#ef4444', width=2),
            name='Perfect Prediction',
            showlegend=True,
        ))
        fig_avp.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            margin=dict(l=20, r=20, t=40, b=20),
            title=dict(text="Actual vs Predicted Prices", font=dict(size=15, color='#c7d2fe')),
            xaxis=dict(title="Actual Price (£)", gridcolor='rgba(99,102,241,0.1)'),
            yaxis=dict(title="Predicted Price (£)", gridcolor='rgba(99,102,241,0.1)'),
            legend=dict(bgcolor='rgba(0,0,0,0.3)', font=dict(color='#c7d2fe')),
        )
        st.plotly_chart(fig_avp, use_container_width=True)
    with ins_col2:
        # Residuals distribution
        residuals = y_test_arr - y_pred_arr
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(
            x=residuals,
            nbinsx=50,
            marker_color='rgba(139,92,246,0.6)',
            marker_line_color='rgba(167,139,250,0.8)',
            marker_line_width=1,
            hovertemplate="Error: £%{x:,.0f}<br>Count: %{y}<extra></extra>",
        ))
        fig_res.add_vline(x=0, line_dash="dash", line_color="#ef4444", line_width=2)
        fig_res.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            margin=dict(l=20, r=20, t=40, b=20),
            title=dict(text="Prediction Error Distribution (Residuals)", font=dict(size=15, color='#c7d2fe')),
            xaxis=dict(title="Error (Actual − Predicted) £", gridcolor='rgba(99,102,241,0.1)'),
            yaxis=dict(title="Frequency", gridcolor='rgba(99,102,241,0.1)'),
        )
        st.plotly_chart(fig_res, use_container_width=True)
    # Feature importance (coefficients)
    st.markdown("""
    <div class="section-header">
        ⚖️ Top Feature Importance (Coefficients)
        <span class="section-header-line"></span>
    </div>
    """, unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.coef_
    })
    coef_df['Abs_Coeff'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coeff', ascending=True).tail(15)
    fig_coef = go.Figure()
    colors = ['#10b981' if c > 0 else '#ef4444' for c in coef_df['Coefficient']]
    fig_coef.add_trace(go.Bar(
        y=coef_df['Feature'].str.replace('model_', '').str.replace('transmission_', '').str.replace('fuelType_', ''),
        x=coef_df['Coefficient'],
        orientation='h',
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Coefficient: %{x:,.0f}<extra></extra>",
    ))
    fig_coef.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text="Top 15 Feature Coefficients (Green = ↑ Price, Red = ↓ Price)", font=dict(size=14, color='#c7d2fe')),
        xaxis=dict(title="Coefficient Value", gridcolor='rgba(99,102,241,0.1)'),
        yaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
    )
    st.plotly_chart(fig_coef, use_container_width=True)
    # Model summary
    st.markdown(f"""
    <div class="info-box">
        <strong>ℹ️ Model Summary</strong><br>
        <strong>Algorithm:</strong> Linear Regression with One-Hot Encoding<br>
        <strong>Features:</strong> {len(feature_cols)} (5 numerical + {len(feature_cols)-5} one-hot encoded categorical)<br>
        <strong>Train/Test Split:</strong> 67% / 33% (random_state=42)<br>
        <strong>Numerical Scaling:</strong> StandardScaler (z-score normalization)
    </div>
    """, unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════
# TAB 4: RAW DATA
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="section-header">
        📋 Dataset Explorer
        <span class="section-header-line"></span>
    </div>
    """, unsafe_allow_html=True)
    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        filter_models = st.multiselect("Filter by Model", options=sorted(df['model'].unique()), default=[])
    with filter_col2:
        filter_trans = st.multiselect("Filter by Transmission", options=sorted(df['transmission'].unique()), default=[])
    with filter_col3:
        price_range = st.slider("Price Range (£)", int(df['price'].min()), int(df['price'].max()), (int(df['price'].min()), int(df['price'].max())))
    filtered_df = df.copy()
    if filter_models:
        filtered_df = filtered_df[filtered_df['model'].isin(filter_models)]
    if filter_trans:
        filtered_df = filtered_df[filtered_df['transmission'].isin(filter_trans)]
    filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]
    st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} records**")
    st.dataframe(
        filtered_df.style.format({'price': '£{:,.0f}', 'mpg': '{:.1f}', 'engineSize': '{:.1f}'}),
        use_container_width=True,
        height=500
    )
    # Download
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name="ford_filtered_data.csv",
        mime="text/csv",
    )
# ─────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <strong>Ford DriveValue</strong> · Built with Streamlit & Scikit-learn<br>
    One-Hot Encoded Linear Regression Model · R² Score: {:.1%}<br>
    <span style="font-size:0.7rem;">Dataset: 17,966 Ford used car listings</span>
</div>
""".format(metrics['r2']), unsafe_allow_html=True)
