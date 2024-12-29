import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px

# Configure app layout
st.set_page_config(
    page_title="Comprehensive Life Expectancy Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    return pd.read_csv('Life-Expectancy-Data-Averaged.csv')

data = load_data()

# Sidebar for navigation
st.sidebar.title("Navigate the App")
pages = [
    "Overview",
    "Life Expectancy by Region",
    "Economic Status Impact",
    "Correlation Analysis (Numeric Columns)",
    "Infant and Adult Mortality",
    "HIV Incidence and Vaccination",
    "Alcohol Consumption Trends",
    "BMI and Health Impact",
    "Population Dynamics",
    "Schooling and Education",
    "Thinness and Nutrition",
    "High Risk Countries",
    "Healthcare Expenditure",
    "Top and Bottom 10 Countries by Life Expectancy",
    "Insights and Recommendations"
]
page = st.sidebar.radio("Select a page:", pages)

#  plotting fuction
def plot_figure(fig):
    st.plotly_chart(fig, use_container_width=True)
# Page: Overview
if page == "Overview":
    st.title("🌍 Overview of Life Expectancy Analysis")
    st.write("This page offers a high-level overview of the dataset and its key attributes.")

    st.header("Dataset Preview")
    st.dataframe(data.head())

    st.header("Summary Statistics Visualization")
    summary_data = data.describe().transpose().reset_index()
    fig = px.bar(summary_data, x='index', y='mean', title="Mean Values of Numerical Columns", labels={'index': 'Column', 'mean': 'Mean Value'})
    plot_figure(fig)

    st.header("Group By Analysis")
    group_options = [
        'Region',  'Adult_mortality', 'Alcohol_consumption', 'BMI',
        'GDP_per_capita', 'Population_mln', 'Schooling', 'Economy_status'
    ]
    group_by_column = st.selectbox("Select a column to group by:", options=group_options)
    if group_by_column:
        grouped_data = data.groupby(group_by_column)['Life_expectancy'].mean().reset_index()
        fig = px.bar(grouped_data, x=group_by_column, y='Life_expectancy', title=f"Life Expectancy Grouped by {group_by_column}")
        plot_figure(fig)
