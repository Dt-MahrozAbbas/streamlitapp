import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

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

# Sidebar navigation with icons
st.sidebar.title("🧭 Navigation Hub")

# Define pages with their icons
pages = {
    "Overview": "🏠",
    "Life Expectancy by Region": "🌍",
    "Infant and Adult Mortality": "👶",
    "Alcohol Consumption Trends": "🍷",
    "BMI and Health Impact": "⚕️",
    "Economy vs Life Expectancy": "💰",
    "Life Expectancy Predicted Model": "🔮"
}

# Create a single radio button group with all options
page_names = list(pages.keys())
page_icons = [pages[name] for name in page_names]
formatted_pages = [f"{icon} {name}" for name, icon in zip(page_names, page_icons)]

selected_index = st.sidebar.radio(
    "",
    range(len(formatted_pages)),
    format_func=lambda x: formatted_pages[x],
    label_visibility="collapsed"
)

# Update the page variable to work with existing code
page = page_names[selected_index]



# Helper function for plotting
def plot_figure(fig):
    st.plotly_chart(fig, use_container_width=True)

# Page: Overview
if page == "Overview":
    st.title("Global Life Expectancy Analysis")
    st.subheader("Exploring worldwide health trends and socioeconomic factors")

    # Dataset Preview
    st.header("Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    # Summary Statistics
    st.header("Summary Statistics")
    st.write(data.describe())

    # Key metrics in columns
    st.header("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_life = data['Life_expectancy'].mean()
        st.metric("Average Life Expectancy", f"{avg_life:.1f} years")
    
    with col2:
        avg_gdp = data['GDP_per_capita'].mean()
        st.metric("Average GDP per Capita", f"${avg_gdp:,.0f}")
    
    with col3:
        avg_schooling = data['Schooling'].mean()
        st.metric("Average Years of Schooling", f"{avg_schooling:.1f}")
    
    with col4:
        countries_count = len(data['Country'].unique())
        st.metric("Countries Analyzed", countries_count)

    # Relationships Analysis
    st.header("Key Variables Relationship Analysis")
    st.write("Exploring relationships between key health and socioeconomic indicators:")
    
    # Select relevant numerical columns for the pairplot
    pairplot_vars = ['Life_expectancy', 'GDP_per_capita', 'Schooling', 
                     'Adult_mortality', 'BMI', 'Alcohol_consumption']
    
    with st.spinner("Generating visualization... This may take a moment."):
        fig = px.scatter_matrix(
            data,
            dimensions=pairplot_vars,
            title="Relationships between Key Variables",
            color='Economy_status',
            opacity=0.7
        )
        fig.update_layout(height=800, width=800)
        fig.update_traces(diagonal_visible=False)
        plot_figure(fig)

# Page: Life Expectancy by Region
elif page == "Life Expectancy by Region":
    st.title("Life Expectancy Analysis by Region")
    
    # Global Overview Section
    st.header("🌍 Global Regional Overview")
    
    # Calculate global region statistics
    region_stats = data.groupby('Region')['Life_expectancy'].agg([
        ('Mean', 'mean'),
        ('Min', 'min'),
        ('Max', 'max')
    ]).round(1)
    
    # Display key global metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        global_avg = data['Life_expectancy'].mean()
        st.metric("Global Average", f"{global_avg:.1f} years")
    with col2:
        highest_region = region_stats['Mean'].idxmax()
        st.metric("Highest Region", f"{highest_region}")
    with col3:
        lowest_region = region_stats['Mean'].idxmin()
        st.metric("Lowest Region", f"{lowest_region}")

    # Global visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot of life expectancy by region
        fig_box = px.box(
            data,
            x='Region',
            y='Life_expectancy',
            color='Region',
            title='Life Expectancy Distribution by Region',
            labels={'Life_expectancy': 'Life Expectancy (years)'}
        )
        fig_box.update_layout(xaxis_tickangle=-45)
        plot_figure(fig_box)
    
    with col2:
        # Bar plot of average life expectancy by region
        fig_avg = px.bar(
            region_stats,
            y='Mean',
            color=region_stats.index,
            title='Average Life Expectancy by Region',
            labels={'Mean': 'Life Expectancy (years)', 'index': 'Region'}
        )
        fig_avg.update_layout(xaxis_tickangle=-45, showlegend=False)
        plot_figure(fig_avg)

    # Regional Analysis Section
    st.header("🔍 Detailed Regional Analysis")

    # Region selection dropdown with "All Regions" option
    selected_region = st.selectbox(
        "Select a region to analyze:",
        options=['All Regions'] + sorted(data['Region'].unique())
    )

    # Filter data based on selection
    region_data = data if selected_region == 'All Regions' else data[data['Region'] == selected_region]

    # Regional metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        region_avg = region_data['Life_expectancy'].mean()
        st.metric("Average", f"{region_avg:.1f} years")
    with col2:
        highest_country = region_data.loc[region_data['Life_expectancy'].idxmax()]
        st.metric("Highest Country", 
                 f"{highest_country['Country']}", 
                 f"{highest_country['Life_expectancy']:.1f} years")
    with col3:
        lowest_country = region_data.loc[region_data['Life_expectancy'].idxmin()]
        st.metric("Lowest Country", 
                 f"{lowest_country['Country']}", 
                 f"{lowest_country['Life_expectancy']:.1f} years")

    # Regional visualizations
    col1, col2 = st.columns(2)

    with col1:
        if selected_region == 'All Regions':
            # Regional averages comparison
            region_avgs = data.groupby('Region')['Life_expectancy'].mean().sort_values(ascending=True)
            fig_top = px.bar(
                region_avgs,
                orientation='h',
                title='Average Life Expectancy by Region',
                labels={'value': 'Life Expectancy (years)', 'Region': 'Region'}
            )
            fig_top.update_layout(height=400)
        else:
            # Top 10 countries in region
            top_10 = region_data.nlargest(10, 'Life_expectancy')
            fig_top = px.bar(
                top_10,
                x='Country',
                y='Life_expectancy',
                title=f'Top 10 Countries in {selected_region}',
                color='Life_expectancy',
                labels={'Life_expectancy': 'Life Expectancy (years)'}
            )
            fig_top.update_layout(xaxis_tickangle=-45)
        plot_figure(fig_top)

    with col2:
        if selected_region == 'All Regions':
            # Life expectancy distribution by region
            fig_bottom = px.violin(
                data,
                x='Region',
                y='Life_expectancy',
                title='Life Expectancy Distribution by Region',
                labels={'Life_expectancy': 'Life Expectancy (years)'}
            )
            fig_bottom.update_layout(xaxis_tickangle=-45)
        else:
            # Bottom 10 countries in region
            bottom_10 = region_data.nsmallest(10, 'Life_expectancy')
            fig_bottom = px.bar(
                bottom_10,
                x='Country',
                y='Life_expectancy',
                title=f'Bottom 10 Countries in {selected_region}',
                color='Life_expectancy',
                labels={'Life_expectancy': 'Life Expectancy (years)'}
            )
            fig_bottom.update_layout(xaxis_tickangle=-45)
        plot_figure(fig_bottom)

    # Additional insights
    st.subheader("📊 Statistical Analysis")

    if selected_region == 'All Regions':
        # Global analysis
        region_stats = data.groupby('Region')['Life_expectancy'].agg([
            ('Mean', 'mean'),
            ('Std', 'std'),
            ('Min', 'min'),
            ('Max', 'max')
        ]).round(1)
        
        st.write("**Regional Comparison:**")
        st.dataframe(region_stats, use_container_width=True)
        
        # Additional global insights
        st.write(f"""
        **Global Insights:**
        - Global average life expectancy: {data['Life_expectancy'].mean():.1f} years
        - Global standard deviation: {data['Life_expectancy'].std():.1f} years
        - Total number of countries: {len(data)}
        - Number of regions: {len(data['Region'].unique())}
        - Life expectancy range: {data['Life_expectancy'].min():.1f} to {data['Life_expectancy'].max():.1f} years
        """)
    else:
        # Regional specific analysis
        regional_spread = region_data['Life_expectancy'].max() - region_data['Life_expectancy'].min()
        global_position = list(region_stats['Mean'].sort_values(ascending=False).index).index(selected_region) + 1
        
        # Create regional stats DataFrame similar to global view
        specific_region_stats = pd.DataFrame({
            'Mean': [region_data['Life_expectancy'].mean()],
            'Min': [region_data['Life_expectancy'].min()],
            'Max': [region_data['Life_expectancy'].max()],
            'Std': [region_data['Life_expectancy'].std()],
            'Global Rank': [global_position],
            'Countries': [len(region_data)],
            'Diff from Global': [(region_avg - global_avg)]
        }, index=[selected_region])
        
        st.write(f"**Statistics for {selected_region}:**")
        st.dataframe(specific_region_stats.round(1), use_container_width=True)

# Page: Infant and Adult Mortality
elif page == "Infant and Adult Mortality":
    st.title("👶 Infant and Adult Mortality Analysis")
    
    # Overview metrics
    st.header("Global Mortality Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        global_adult_mortality = data['Adult_mortality'].mean()
        st.metric("Global Adult Mortality Rate", f"{global_adult_mortality:.1f}")
    with col2:
        global_infant_deaths = data['Infant_deaths'].mean()
        st.metric("Global Infant Deaths", f"{global_infant_deaths:.1f}")
    with col3:
        highest_adult = data.groupby('Region')['Adult_mortality'].mean().max()
        st.metric("Highest Regional Adult Mortality", f"{highest_adult:.1f}")
    with col4:
        highest_infant = data.groupby('Region')['Infant_deaths'].mean().max()
        st.metric("Highest Regional Infant Deaths", f"{highest_infant:.1f}")

    # Interactive Region Selection
    st.subheader("🔍 Region Selection")
    selected_region = st.selectbox(
        "Choose a region to analyze:",
        options=['All Regions'] + sorted(data['Region'].unique().tolist())
    )
    
    filtered_data = data if selected_region == 'All Regions' else data[data['Region'] == selected_region]

    # Main Analysis Section
    st.header("📊 Mortality Analysis")
    
    # Mortality by Region
    col1, col2 = st.columns(2)
    
    with col1:
        fig_adult = px.box(
            data,
            x='Region',
            y='Adult_mortality',
            color='Region',
            title='Adult Mortality Distribution by Region',
            labels={'Adult_mortality': 'Adult Mortality Rate'}
        )
        fig_adult.update_layout(xaxis_tickangle=-45)
        plot_figure(fig_adult)
    
    with col2:
        fig_infant = px.box(
            data,
            x='Region',
            y='Infant_deaths',
            color='Region',
            title='Infant Deaths Distribution by Region',
            labels={'Infant_deaths': 'Infant Deaths'}
        )
        fig_infant.update_layout(xaxis_tickangle=-45)
        plot_figure(fig_infant)

    # Economic Impact
    st.header("💰 Economic Status Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_eco_adult = px.violin(
            filtered_data,
            x='Economy_status',
            y='Adult_mortality',
            color='Economy_status',
            box=True,
            title='Adult Mortality by Economic Status'
        )
        plot_figure(fig_eco_adult)
    
    with col2:
        fig_eco_infant = px.violin(
            filtered_data,
            x='Economy_status',
            y='Infant_deaths',
            color='Economy_status',
            box=True,
            title='Infant Deaths by Economic Status'
        )
        plot_figure(fig_eco_infant)

    # Under-Five Deaths Analysis
    st.header("👶 Under-Five Deaths Analysis")
    
    fig_under_five = px.scatter(
        filtered_data,
        x='Infant_deaths',
        y='Under_five_deaths',
        color='Economy_status',
        size='GDP_per_capita',
        hover_data=['Country', 'Life_expectancy'],
        title='Infant Deaths vs Under-Five Deaths',
        labels={
            'Infant_deaths': 'Infant Deaths',
            'Under_five_deaths': 'Under-Five Deaths'
        }
    )
    plot_figure(fig_under_five)

    # Impact on Life Expectancy
    st.header("⏳ Impact on Life Expectancy")
    
    # 3D scatter plot
    fig_3d = px.scatter_3d(
        filtered_data,
        x='Adult_mortality',
        y='Infant_deaths',
        z='Life_expectancy',
        color='Economy_status',
        size='GDP_per_capita',
        hover_data=['Country'],
        title='Mortality Rates and Life Expectancy',
        labels={
            'Adult_mortality': 'Adult Mortality Rate',
            'Infant_deaths': 'Infant Deaths',
            'Life_expectancy': 'Life Expectancy (years)'
        }
    )
    fig_3d.update_layout(height=800)
    plot_figure(fig_3d)

    # Risk Analysis
    st.header("⚠️ High-Risk Analysis")
    
    # Calculate thresholds (75th percentile)
    adult_threshold = data['Adult_mortality'].quantile(0.75)
    infant_threshold = data['Infant_deaths'].quantile(0.75)
    
    high_risk = data[
        (data['Adult_mortality'] > adult_threshold) & 
        (data['Infant_deaths'] > infant_threshold)
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("High-Risk Countries")
        risk_by_region = high_risk.groupby('Region').size().sort_values(ascending=False)
        fig_risk = px.bar(
            risk_by_region,
            title='Number of High-Risk Countries by Region',
            labels={'value': 'Number of Countries', 'Region': 'Region'}
        )
        plot_figure(fig_risk)
    
    with col2:
        st.subheader("Risk Factors")
        risk_factors = high_risk[['GDP_per_capita', 'Schooling', 'BMI']].mean()
        safe_factors = data[
            (data['Adult_mortality'] <= adult_threshold) & 
            (data['Infant_deaths'] <= infant_threshold)
        ][['GDP_per_capita', 'Schooling', 'BMI']].mean()
        
        comparison = pd.DataFrame({
            'High Risk': risk_factors,
            'Low Risk': safe_factors
        }).round(2)
        st.dataframe(comparison)

    # Key Findings
    st.header("🔍 Key Findings")
    
    # Calculate statistics
    mortality_corr = data['Adult_mortality'].corr(data['Infant_deaths'])
    life_exp_corr_adult = data['Life_expectancy'].corr(data['Adult_mortality'])
    life_exp_corr_infant = data['Life_expectancy'].corr(data['Infant_deaths'])
    
    st.write(f"""
    Key observations about mortality rates:
    
    - **Correlation Analysis**:
        - Adult Mortality and Infant Deaths Correlation: {mortality_corr:.3f}
        - Life Expectancy correlation with:
            - Adult Mortality: {life_exp_corr_adult:.3f}
            - Infant Deaths: {life_exp_corr_infant:.3f}
    
    - **Regional Patterns**:
        - Highest Adult Mortality: {data.groupby('Region')['Adult_mortality'].mean().idxmax()}
        - Highest Infant Deaths: {data.groupby('Region')['Infant_deaths'].mean().idxmax()}
    
    - **Economic Impact**: 
        - Strong relationship between economic status and mortality rates
        - Developed countries show significantly lower mortality rates
    
    - **High-Risk Areas**:
        - {len(high_risk)} countries show high risk in both mortality metrics
        - These countries have significantly lower GDP and schooling rates
    """)

# Page: BMI and Health Impact
elif page == "BMI and Health Impact":
    st.title("🏃‍♂️ BMI and Health Impact Analysis")
    
    # Overview metrics
    st.header("Global BMI Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        global_bmi_mean = data['BMI'].mean()
        st.metric("Global Average BMI", f"{global_bmi_mean:.1f}")
    with col2:
        highest_bmi = data.groupby('Region')['BMI'].mean().max()
        st.metric("Highest Regional BMI", f"{highest_bmi:.1f}")
    with col3:
        lowest_bmi = data.groupby('Region')['BMI'].mean().min()
        st.metric("Lowest Regional BMI", f"{lowest_bmi:.1f}")

    # BMI Distribution
    st.header("📊 BMI Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # BMI Distribution by Region
        fig_bmi_region = px.box(
            data,
            x='Region',
            y='BMI',
            color='Region',
            title='BMI Distribution by Region'
        )
        fig_bmi_region.update_layout(xaxis_tickangle=-45)
        plot_figure(fig_bmi_region)
    
    with col2:
        # BMI vs Economic Status
        fig_bmi_economy = px.violin(
            data,
            x='Economy_status',
            y='BMI',
            color='Economy_status',
            box=True,
            title='BMI Distribution by Economic Status'
        )
        plot_figure(fig_bmi_economy)

    # BMI and Life Expectancy Relationship
    st.header("🔍 BMI Impact on Life Expectancy")
    
    # Scatter plot without trend line
    fig_bmi_life = px.scatter(
        data,
        x='BMI',
        y='Life_expectancy',
        color='Economy_status',
        title='BMI vs Life Expectancy',
        labels={'BMI': 'Body Mass Index', 'Life_expectancy': 'Life Expectancy (years)'}
    )
    plot_figure(fig_bmi_life)

    # BMI Trends
    st.header("📈 BMI Trends and Correlations")
    
    # Correlation heatmap
    correlation_vars = ['BMI', 'Life_expectancy', 'GDP_per_capita', 
                       'Adult_mortality', 'Alcohol_consumption']
    correlation_matrix = data[correlation_vars].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        title='Correlation between BMI and Other Factors',
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    plot_figure(fig_corr)

    # Additional insights
    st.header("📋 Key Findings")
    
    # Calculate some statistics
    high_bmi_life = data[data['BMI'] > data['BMI'].median()]['Life_expectancy'].mean()
    low_bmi_life = data[data['BMI'] <= data['BMI'].median()]['Life_expectancy'].mean()
    bmi_life_corr = data['BMI'].corr(data['Life_expectancy'])
    
    st.write(f"""
    Key observations about BMI and health:
    
    - **Average Life Expectancy**:
        - Countries with above-median BMI: {high_bmi_life:.1f} years
        - Countries with below-median BMI: {low_bmi_life:.1f} years
    
    - **Correlation with Life Expectancy**: {bmi_life_corr:.3f}
    
    - **Regional Patterns**:
        - Highest average BMI is found in {data.groupby('Region')['BMI'].mean().idxmax()}
        - Lowest average BMI is found in {data.groupby('Region')['BMI'].mean().idxmin()}
    
    - **Economic Impact**: There appears to be a relationship between economic status and BMI levels,
      with developed countries generally showing higher BMI values.
    """)

# Page: Alcohol Consumption Trends
elif page == "Alcohol Consumption Trends":
    st.title("🍷 Global Alcohol Consumption Analysis")
    
    # Global Analysis Section
    st.header("🌍 Global Alcohol Consumption Patterns")
    
    # Global metrics in two rows
    col1, col2 = st.columns(2)
    with col1:
        # Top 10 consuming countries
        top_consumers = data.nlargest(10, 'Alcohol_consumption')
        fig_top = px.bar(
            top_consumers,
            x='Country',
            y='Alcohol_consumption',
            color='Region',
            title='Highest Alcohol Consuming Countries',
            labels={'Alcohol_consumption': 'Alcohol Consumption (L)'}
        )
        fig_top.update_layout(xaxis_tickangle=-45)
        plot_figure(fig_top)
        
    with col2:
        # Bottom 10 consuming countries
        bottom_consumers = data.nsmallest(10, 'Alcohol_consumption')
        fig_bottom = px.bar(
            bottom_consumers,
            x='Country',
            y='Alcohol_consumption',
            color='Region',
            title='Lowest Alcohol Consuming Countries',
            labels={'Alcohol_consumption': 'Alcohol Consumption (L)'}
        )
        fig_bottom.update_layout(xaxis_tickangle=-45)
        plot_figure(fig_bottom)

    # Global statistics
    st.subheader("Global Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        global_mean = data['Alcohol_consumption'].mean()
        st.metric("Global Average", f"{global_mean:.1f}L")
    with col2:
        global_max = data['Alcohol_consumption'].max()
        max_country = data.loc[data['Alcohol_consumption'].idxmax(), 'Country']
        st.metric("Highest Consumption", f"{global_max:.1f}L", f"({max_country})")
    with col3:
        global_min = data['Alcohol_consumption'].min()
        min_country = data.loc[data['Alcohol_consumption'].idxmin(), 'Country']
        st.metric("Lowest Consumption", f"{global_min:.1f}L", f"({min_country})")

    # Regional Analysis Section
    st.header("🗺️ Regional Analysis")

    # Region selection
    selected_region = st.selectbox(
        "Select a region to analyze:",
        options=['All Regions'] + sorted(data['Region'].unique().tolist())
    )

    # Filter data based on selection
    filtered_data = data if selected_region == 'All Regions' else data[data['Region'] == selected_region]

    # Calculate life expectancy statistics for both high and low alcohol consumption
    high_alcohol_life = filtered_data[
        filtered_data['Alcohol_consumption'] > filtered_data['Alcohol_consumption'].median()
    ]['Life_expectancy'].mean()
    low_alcohol_life = filtered_data[
        filtered_data['Alcohol_consumption'] <= filtered_data['Alcohol_consumption'].median()
    ]['Life_expectancy'].mean()

    # Regional visualizations
    if selected_region != 'All Regions':
        st.subheader(f"Detailed Analysis for {selected_region}")
        
        col1, col2 = st.columns(2)
        with col1:
            # Top 5 countries in region
            top_regional = filtered_data.nlargest(5, 'Alcohol_consumption')
            fig_top_regional = px.bar(
                top_regional,
                x='Country',
                y='Alcohol_consumption',
                title=f'Top 5 Consuming Countries in {selected_region}',
                color='Life_expectancy',
                color_continuous_scale='RdYlBu',
                labels={'Life_expectancy': 'Life Expectancy (years)'}
            )
            plot_figure(fig_top_regional)
            
        with col2:
            # Scatter plot of alcohol vs life expectancy
            fig_life = px.scatter(
                filtered_data,
                x='Alcohol_consumption',
                y='Life_expectancy',
                color='GDP_per_capita',
                size='Adult_mortality',
                hover_data=['Country'],
                title='Alcohol Consumption vs Life Expectancy',
                labels={
                    'Alcohol_consumption': 'Alcohol Consumption (L)',
                    'Life_expectancy': 'Life Expectancy (years)',
                    'GDP_per_capita': 'GDP per Capita'
                }
            )
            plot_figure(fig_life)
        
        # Regional statistics
        st.subheader("Regional Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            regional_mean = filtered_data['Alcohol_consumption'].mean()
            st.metric("Regional Average", f"{regional_mean:.1f}L")
        with col2:
            st.metric("Life Expectancy (High Consumption)", f"{high_alcohol_life:.1f} years")
        with col3:
            st.metric("Life Expectancy (Low Consumption)", f"{low_alcohol_life:.1f} years")
    
    else:
        # Show global regional comparison
        col1, col2 = st.columns(2)
        with col1:
            # Regional averages
            regional_avg = data.groupby('Region')['Alcohol_consumption'].mean().sort_values(ascending=False)
            fig_avg = px.bar(
                regional_avg,
                title='Average Alcohol Consumption by Region',
                labels={'value': 'Average Consumption (L)', 'Region': 'Region'}
            )
            plot_figure(fig_avg)
        
        with col2:
            # Regional life expectancy correlation
            fig_region_life = px.box(
                data,
                x='Region',
                y='Life_expectancy',
                color='Region',
                title='Life Expectancy Distribution by Region',
                labels={'Life_expectancy': 'Life Expectancy (years)'}
            )
            fig_region_life.update_layout(xaxis_tickangle=-45)
            plot_figure(fig_region_life)

    # Key insights
    st.header("🔍 Key Insights")
    alcohol_life_corr = filtered_data['Alcohol_consumption'].corr(filtered_data['Life_expectancy'])
    
    insights = f"""
    {'Regional' if selected_region != 'All Regions' else 'Global'} Analysis Shows:
    
    - **Consumption Patterns**:
        - {'Regional' if selected_region != 'All Regions' else 'Global'} Average: {filtered_data['Alcohol_consumption'].mean():.1f}L
        - Highest Consumer: {filtered_data.loc[filtered_data['Alcohol_consumption'].idxmax(), 'Country']} ({filtered_data['Alcohol_consumption'].max():.1f}L)
        - Lowest Consumer: {filtered_data.loc[filtered_data['Alcohol_consumption'].idxmin(), 'Country']} ({filtered_data['Alcohol_consumption'].min():.1f}L)
    
    - **Health Correlations**:
        - Correlation with Life Expectancy: {alcohol_life_corr:.3f}
        - Life Expectancy Difference (High vs Low Consumption): {high_alcohol_life - low_alcohol_life:.1f} years
    """
    st.write(insights)  
# Page: GDP and Economy vs Life Expectancy
elif page == "Economy vs Life Expectancy":
    st.title("💰 GDP, Economy and Life Expectancy Analysis")
    
    # Global Analysis Section
    st.header("🌍 Global Economic Impact on Life Expectancy")
    
    # Global statistics in three columns
    col1, col2, col3 = st.columns(3)
    with col1:
        global_gdp_mean = data['GDP_per_capita'].mean()
        st.metric("Global Average GDP", f"${global_gdp_mean:,.0f}")
    with col2:
        high_gdp_life = data[data['GDP_per_capita'] > data['GDP_per_capita'].median()]['Life_expectancy'].mean()
        st.metric("Life Expectancy (High GDP)", f"{high_gdp_life:.1f} years")
    with col3:
        low_gdp_life = data[data['GDP_per_capita'] <= data['GDP_per_capita'].median()]['Life_expectancy'].mean()
        st.metric("Life Expectancy (Low GDP)", f"{low_gdp_life:.1f} years")

    # Global visualizations
    col1, col2 = st.columns(2)
    with col1:
        # GDP vs Life Expectancy scatter plot
        fig_gdp_life = px.scatter(
            data,
            x='GDP_per_capita',
            y='Life_expectancy',
            color='Economy_status',
            title='GDP per Capita vs Life Expectancy',
            labels={
                'GDP_per_capita': 'GDP per Capita ($)',
                'Life_expectancy': 'Life Expectancy (years)'
            },
            hover_data=['Country']
        )
        plot_figure(fig_gdp_life)
    
    with col2:
        # Life expectancy by economic status
        fig_economy = px.box(
            data,
            x='Economy_status',
            y='Life_expectancy',
            color='Economy_status',
            title='Life Expectancy Distribution by Economic Status',
            labels={'Life_expectancy': 'Life Expectancy (years)'}
        )
        plot_figure(fig_economy)

    # Regional Analysis Section
    st.header("🗺️ Regional Economic Analysis")

    # Region selection
    selected_region = st.selectbox(
        "Select Region for Detailed Analysis:",
        ['All Regions'] + list(data['Region'].unique())
    )

    # Filter data based on selection
    filtered_data = data if selected_region == 'All Regions' else data[data['Region'] == selected_region]

    # Regional statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        regional_gdp_mean = filtered_data['GDP_per_capita'].mean()
        st.metric("Regional Average GDP", f"${regional_gdp_mean:,.0f}")
    with col2:
        highest_gdp_country = filtered_data.loc[filtered_data['GDP_per_capita'].idxmax(), 'Country']
        highest_gdp = filtered_data['GDP_per_capita'].max()
        st.metric("Highest GDP", f"${highest_gdp:,.0f}", f"({highest_gdp_country})")
    with col3:
        lowest_gdp_country = filtered_data.loc[filtered_data['GDP_per_capita'].idxmin(), 'Country']
        lowest_gdp = filtered_data['GDP_per_capita'].min()
        st.metric("Lowest GDP", f"${lowest_gdp:,.0f}", f"({lowest_gdp_country})")

    # Regional visualizations
    col1, col2 = st.columns(2)
    with col1:
        # Top 10 GDP countries in region
        top_gdp = filtered_data.nlargest(10, 'GDP_per_capita')
        fig_top_gdp = px.bar(
            top_gdp,
            x='Country',
            y='GDP_per_capita',
            color='Economy_status',
            title=f'Top 10 Countries by GDP per Capita ({selected_region})',
            labels={'GDP_per_capita': 'GDP per Capita ($)'}
        )
        fig_top_gdp.update_layout(xaxis_tickangle=-45)
        plot_figure(fig_top_gdp)
    
    with col2:
        # GDP vs Life Expectancy in region
        fig_region_gdp = px.scatter(
            filtered_data,
            x='GDP_per_capita',
            y='Life_expectancy',
            color='Economy_status',
            title=f'GDP vs Life Expectancy in {selected_region}',
            labels={
                'GDP_per_capita': 'GDP per Capita ($)',
                'Life_expectancy': 'Life Expectancy (years)'
            },
            hover_data=['Country']
        )
        plot_figure(fig_region_gdp)

    # Key insights
    st.header("📊 Economic Insights")
    gdp_life_corr = filtered_data['GDP_per_capita'].corr(filtered_data['Life_expectancy'])
    
    insights = f"""
    {'Regional' if selected_region != 'All Regions' else 'Global'} Analysis Shows:
    
    - **Economic Indicators**:
        - Average GDP per Capita: ${filtered_data['GDP_per_capita'].mean():,.0f}
        - GDP Range: ${filtered_data['GDP_per_capita'].min():,.0f} to ${filtered_data['GDP_per_capita'].max():,.0f}
    
    - **Health Correlations**:
        - Correlation between GDP and Life Expectancy: {gdp_life_corr:.3f}
        - Life Expectancy Gap (High vs Low GDP): {high_gdp_life - low_gdp_life:.1f} years
    
    - **Economic Distribution**:
        - Developed Economies: {len(filtered_data[filtered_data['Economy_status'] == 'Developed'])} countries
        - Developing Economies: {len(filtered_data[filtered_data['Economy_status'] == 'Developing'])} countries
    """
    st.write(insights)
# Page: Life Expectancy Prediction Model
elif page == "Life Expectancy Predicted Model":
    
    st.title("🔮 Enhanced Life Expectancy Prediction Model")
    
    try:
        # Show correlation with Life Expectancy
        st.header("📊 Feature Correlations Analysis")
        
        # Select numerical columns and ensure they exist in the dataset
        numerical_cols = [col for col in ['Life_expectancy', 'Adult_mortality', 
                                        'Alcohol_consumption', 'BMI', 
                                        'GDP_per_capita', 'Schooling'] 
                         if col in data.columns]
        
        # Check if we have enough features
        if len(numerical_cols) < 2:
            st.error("Not enough numerical features available in the dataset")
            st.stop()
            
        correlation_matrix = data[numerical_cols].corr()
        
        # Plot correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            title='Feature Correlation Heatmap',
            color_continuous_scale='RdBu',
            aspect='auto',
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(
            xaxis_title="Features",
            yaxis_title="Features"
        )
        plot_figure(fig_corr)
        
        # Select features and verify they exist in the dataset
        selected_features = ['Adult_mortality', 'Schooling', 'GDP_per_capita', 'BMI']
        if not all(feature in data.columns for feature in selected_features):
            st.error("Some required features are missing from the dataset")
            st.stop()
        
        st.write("""
        Based on correlation analysis, we selected the following key features for prediction:
        - Adult Mortality Rate (Strong negative correlation)
        - Years of Schooling (Strong positive correlation)
        - GDP per Capita (Strong positive correlation)
        - BMI (Body Mass Index) (Moderate positive correlation)
        """)
        
        # Prepare data
        X = data[selected_features].copy()  # Use copy to avoid SettingWithCopyWarning
        y = data['Life_expectancy'].copy()
        
        # Check for and handle missing values
        if X.isnull().any().any() or y.isnull().any():
            st.warning("Handling missing values in the dataset...")
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and fit scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Display metrics
        st.header("📈 Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        with col2:
            st.metric("RMSE", f"{rmse:.2f} years")
        with col3:
            st.metric("MAE", f"{mae:.2f} years")
        with col4:
            st.metric("Training Size", f"{len(X_train)} samples")
        
        st.write("""
        **Interpretation of Metrics:**
        - R² Score: Indicates how well the model fits the data (1.0 is perfect fit)
        - RMSE: Average prediction error in years
        - MAE: Average absolute prediction error in years
        """)
        
        # Input form
        st.header("🎯 Predict Life Expectancy")
        st.write("Enter values for the features below to get a life expectancy prediction")
        
        # Get min and max values from the dataset for better input validation
        col1, col2 = st.columns(2)
        
        with col1:
            adult_mortality = st.number_input(
                "Adult Mortality Rate (per 1000)", 
                min_value=float(X['Adult_mortality'].min()),
                max_value=float(X['Adult_mortality'].max()),
                value=float(X['Adult_mortality'].median())
            )
            schooling = st.number_input(
                "Years of Schooling",
                min_value=float(X['Schooling'].min()),
                max_value=float(X['Schooling'].max()),
                value=float(X['Schooling'].median())
            )
        
        with col2:
            gdp = st.number_input(
                "GDP per Capita ($)",
                min_value=float(X['GDP_per_capita'].min()),
                max_value=float(X['GDP_per_capita'].max()),
                value=float(X['GDP_per_capita'].median())
            )
            bmi = st.number_input(
                "Average BMI",
                min_value=float(X['BMI'].min()),
                max_value=float(X['BMI'].max()),
                value=float(X['BMI'].median())
            )
        
        if st.button("Predict Life Expectancy"):
            try:
                # Prepare input data
                input_data = np.array([[adult_mortality, schooling, gdp, bmi]])
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                
                # Validate prediction
                if prediction < 0 or prediction > 100:
                    st.error("Invalid prediction value. Please check input values.")
                else:
                    st.success(f"Predicted Life Expectancy: {prediction:.1f} years")
                
                st.write("""
                **Note**: This is a simplified prediction model based on historical data. 
                Actual life expectancy depends on many more factors including:
                - Healthcare quality and accessibility
                - Environmental conditions
                - Lifestyle choices
                - Genetic factors
                """)
                
                # Show feature importance
                st.header("📊 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': np.abs(model.coef_)
                })
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                fig_importance = px.bar(
                    importance_df,
                    x='Feature',
                    y='Importance',
                    title='Feature Importance in Prediction'
                )
                plot_figure(fig_importance)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your data and try again.")