import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="COVID-19 Data Analytics Dashboard", layout="wide", page_icon="🦠")

# --- CUSTOM HTML/CSS KPI CARD FUNCTION ---
def create_kpi_card(title, value, icon, border_color):
    """Generates an HTML string for a beautiful KPI card."""
    html = f"""
    <div style="background-color: rgba(150, 150, 150, 0.1); 
                padding: 20px; 
                border-radius: 10px; 
                border-left: 8px solid {border_color}; 
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); 
                text-align: center;
                transition: transform 0.2s ease-in-out;">
        <p style="font-size: 16px; margin: 0; padding: 0; font-weight: 600; opacity: 0.8;">{icon} {title}</p>
        <p style="font-size: 32px; margin: 10px 0 0 0; padding: 0; font-weight: bold;">{value}</p>
    </div>
    """
    return html

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_data():
    df = pd.read_csv("covid19_cleaned_dataset.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.fillna(0)
    return df

df = load_data()

# --- SIDEBAR: THEME SETTINGS ---
st.sidebar.header("🎨 Theme Settings")
theme_options = ["plotly_dark", "plotly_white", "ggplot2", "seaborn", "simple_white", "presentation"]
selected_theme = st.sidebar.selectbox("Select Chart Theme", theme_options, index=0)

st.sidebar.divider()

# --- SIDEBAR: FILTERS ---
st.sidebar.header("🔍 Global Filters")

# Filter by Region first to narrow down countries
available_regions = sorted(df['region'].unique())
selected_regions = st.sidebar.multiselect("Select Regions", options=available_regions, default=available_regions)

# Filter countries based on selected regions
filtered_by_region = df[df['region'].isin(selected_regions)]
available_countries = sorted(filtered_by_region['country'].unique())
default_countries = available_countries[:5] if len(available_countries) >= 5 else available_countries
selected_countries = st.sidebar.multiselect("Select Countries", options=available_countries, default=default_countries)

# Date Filter
min_date = df['date'].min().date()
max_date = df['date'].max().date()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# Apply All Filters
filtered_df = df[
    (df['region'].isin(selected_regions)) &
    (df['country'].isin(selected_countries)) & 
    (df['date'].dt.date >= start_date) & 
    (df['date'].dt.date <= end_date)
]

# --- MAIN DASHBOARD HEADER ---
st.title("🦠 COVID-19 Analytics & Prediction")
st.markdown("Interactive visualizations and predictive modeling based on global pandemic metrics.")
st.divider()

if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust your sidebar parameters.")
else:
    # --- BEAUTIFUL KPI CARDS ---
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    total_cases = int(filtered_df['daily_new_cases'].sum())
    total_recovered = int(filtered_df['daily_recovered'].sum())
    total_deaths = int(filtered_df['daily_deaths'].sum())
    total_tests = int(filtered_df['tests_conducted'].sum())
    
    # Inject Custom HTML for Row 1
    col1.markdown(create_kpi_card("Total New Cases", f"{total_cases:,}", "📈", "#FF4B4B"), unsafe_allow_html=True)
    col2.markdown(create_kpi_card("Total Recovered", f"{total_recovered:,}", "🛡️", "#00C853"), unsafe_allow_html=True)
    col3.markdown(create_kpi_card("Total Deaths", f"{total_deaths:,}", "⚠️", "#9C27B0"), unsafe_allow_html=True)
    col4.markdown(create_kpi_card("Tests Conducted", f"{total_tests:,}", "🧪", "#29B6F6"), unsafe_allow_html=True)

    st.write("") # Add a little vertical spacing between rows

    col5, col6, col7, col8 = st.columns(4)
    avg_positivity = filtered_df['positivity_rate'].mean() * 100
    avg_vax_rate = filtered_df['vaccination_rate'].mean() * 100
    avg_stringency = filtered_df['stringency_index'].mean()
    avg_reproduction = filtered_df['reproduction_number'].mean()

    # Inject Custom HTML for Row 2
    col5.markdown(create_kpi_card("Avg Positivity Rate", f"{avg_positivity:.2f}%", "📊", "#FFA726"), unsafe_allow_html=True)
    col6.markdown(create_kpi_card("Avg Vaccination Rate", f"{avg_vax_rate:.2f}%", "💉", "#66BB6A"), unsafe_allow_html=True)
    col7.markdown(create_kpi_card("Avg Stringency Index", f"{avg_stringency:.1f}", "🏛️", "#AB47BC"), unsafe_allow_html=True)
    col8.markdown(create_kpi_card("Avg R0 (Spread)", f"{avg_reproduction:.2f}", "🔄", "#EF5350"), unsafe_allow_html=True)
    
    st.divider()

    # --- INTERACTIVE VISUALIZATIONS (WITH DYNAMIC THEMES) ---
    st.subheader("📈 Interactive Data Explorations")

    # 1. Geographic Map
    st.markdown("#### Global Spread Map")
    map_data = filtered_df.groupby('country')['daily_new_cases'].sum().reset_index()
    fig_map = px.choropleth(map_data, locations="country", locationmode="country names",
                            color="daily_new_cases", hover_name="country",
                            color_continuous_scale="Reds", title="Total Cases by Country",
                            template=selected_theme)
    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    # 2. NEW: Top 10 Countries - Cases vs Deaths
    st.markdown("#### Top 10 Countries: Cases vs Deaths")
    top_10_df = filtered_df.groupby('country')[['daily_new_cases', 'daily_deaths']].sum().reset_index()
    top_10_df = top_10_df.sort_values(by='daily_new_cases', ascending=False).head(10)
    
    if not top_10_df.empty:
        fig_top10 = px.bar(top_10_df, x='country', y=['daily_new_cases', 'daily_deaths'],
                           barmode='group',
                           title="Highest Impact: Total Cases vs Total Deaths (Top 10)",
                           labels={'value': 'Count', 'variable': 'Metric', 'country': 'Country'},
                           template=selected_theme)
        
        # Make the legend labels cleaner
        legend_names = {'daily_new_cases': 'Total Cases', 'daily_deaths': 'Total Deaths'}
        fig_top10.for_each_trace(lambda t: t.update(name = legend_names.get(t.name, t.name),
                                                    legendgroup = legend_names.get(t.name, t.name),
                                                    hovertemplate = t.hovertemplate.replace(t.name, legend_names.get(t.name, t.name))))
        
        st.plotly_chart(fig_top10, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    
    # 3. Area Chart
    with row2_col1:
        st.markdown("#### Trend: Active vs Recovered Cases")
        area_data = filtered_df.groupby('date')[['active_cases', 'daily_recovered']].sum().reset_index()
        fig_area = px.area(area_data, x="date", y=["active_cases", "daily_recovered"], 
                           title="Active vs Recovered Over Time",
                           labels={'value': 'Number of Cases', 'variable': 'Metric', 'date': 'Date'},
                           template=selected_theme)
        st.plotly_chart(fig_area, use_container_width=True)

    # 4. Sunburst Chart
    with row2_col2:
        st.markdown("#### Case Distribution (Region > Country > Variant)")
        sunburst_data = filtered_df[filtered_df['daily_new_cases'] > 0]
        fig_sunburst = px.sunburst(sunburst_data, path=['region', 'country', 'variant'], values='daily_new_cases',
                                   title="Hierarchical Breakdown of Cases",
                                   template=selected_theme)
        st.plotly_chart(fig_sunburst, use_container_width=True)

    row3_col1, row3_col2 = st.columns(2)

    # 5. Scatter Plot
    with row3_col1:
        st.markdown("#### Does Strict Policy Lower R0?")
        fig_scatter2 = px.scatter(filtered_df, x='stringency_index', y='reproduction_number', 
                                  color='country', hover_data=['date'],
                                  title="Stringency Index vs Reproduction Number (R0)",
                                  labels={'stringency_index': 'Govt Stringency Index', 'reproduction_number': 'Reproduction (R0)'},
                                  template=selected_theme)
        st.plotly_chart(fig_scatter2, use_container_width=True)

    # 6. Bar Chart
    with row3_col2:
        st.markdown("#### Impact by Variant")
        variant_df = filtered_df.groupby('variant')[['daily_new_cases']].sum().reset_index()
        fig_bar = px.bar(variant_df, x='variant', y='daily_new_cases', color='variant',
                         title="Total Cases Grouped by Variant",
                         labels={'daily_new_cases': 'Total Cases', 'variant': 'Variant Type'},
                         template=selected_theme)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    # 7. Correlation Heatmap
    st.markdown("#### Feature Correlation Matrix")
    numeric_df = filtered_df.select_dtypes(include=['number'])
    
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                             color_continuous_scale="RdBu_r",
                             title="Correlation Between Numeric Features",
                             template=selected_theme)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric data to generate a correlation heatmap.")

    st.divider()

# --- MACHINE LEARNING MODELS ---
st.subheader("🤖 Predictive Modeling: Daily Cases")
st.markdown("Choose an algorithm below to predict daily new (confirmed) cases based on pandemic metrics.")

@st.cache_resource
def train_models(data):
    features = ['positivity_rate', 'tests_conducted', 'reproduction_number', 'stringency_index', 'vaccination_rate']
    train_df = data.copy()
    X = train_df[features]
    y = train_df['daily_new_cases']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_score = rf_model.score(X_test, y_test)
    
    # Train XGBoost
    xgb_model = XGBRegressor(n_estimators=50, random_state=42, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)
    xgb_score = xgb_model.score(X_test, y_test)
    
    return rf_model, rf_score, xgb_model, xgb_score

# Load both models
rf_model, rf_score, xgb_model, xgb_score = train_models(df)

# Model Selection UI
model_choice = st.radio("Select Prediction Model:", ("Random Forest", "XGBoost"), horizontal=True)

if model_choice == "Random Forest":
    selected_model = rf_model
    st.success(f"Using **Random Forest Regressor**. Current Model Accuracy (R² Score): **{rf_score:.2%}**")
else:
    selected_model = xgb_model
    st.success(f"Using **XGBoost Regressor**. Current Model Accuracy (R² Score): **{xgb_score:.2%}**")

# Prediction Input UI
st.markdown("#### Adjust Parameters to Predict Cases")
pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    in_pos_rate = st.slider("Positivity Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    in_tests = st.number_input("Tests Conducted (Daily)", min_value=0, max_value=2000000, value=50000)

with pred_col2:
    in_rep_num = st.slider("Reproduction Number (R0)", min_value=0.0, max_value=5.0, value=1.2, step=0.1)
    in_stringency = st.slider("Stringency Index (0-100)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)

with pred_col3:
    in_vax_rate = st.slider("Vaccination Rate", min_value=0.0, max_value=1.0, value=0.4, step=0.01)

# Make Prediction
if st.button(f"Predict with {model_choice}", type="primary"):
    input_data = pd.DataFrame({
        'positivity_rate': [in_pos_rate],
        'tests_conducted': [in_tests],
        'reproduction_number': [in_rep_num],
        'stringency_index': [in_stringency],
        'vaccination_rate': [in_vax_rate]
    })
    
    prediction = selected_model.predict(input_data)[0]
    
    # Ensure prediction isn't negative
    final_prediction = max(0, int(prediction))
    
    st.metric(label=f"Predicted Daily New Cases ({model_choice})", value=f"{final_prediction:,}")