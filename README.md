
## 📌 Project Description
This project focuses on building an interactive dashboard for analyzing global life expectancy trends and their relationships with various health, socioeconomic, and environmental factors. Using **Streamlit**, **Plotly**, and **Scikit-learn**, the application provides insights into:

- Regional life expectancy differences.
- Key health indicators like BMI, alcohol consumption, and mortality rates.
- The economic impact on life expectancy.
- Predictive modeling of life expectancy.

---

## 🔧 Features

### 1. **Interactive Dashboard**
Navigate through the following key pages:
- **Overview:** High-level summary of global trends and dataset statistics.
- **Life Expectancy by Region:** Insights into regional variations.
- **Infant and Adult Mortality:** Detailed analysis of mortality trends.
- **Alcohol Consumption Trends:** Patterns and impacts of alcohol use.
- **BMI and Health Impact:** BMI’s role in life expectancy.
- **Economy vs Life Expectancy:** Explore GDP and socioeconomic impacts.
- **Life Expectancy Predicted Model:** Predict life expectancy based on input factors.

### 2. **Data Visualizations**
- Dynamic scatter plots, heatmaps, and violin plots using **Plotly**.
- Interactive widgets for region and feature selection.

### 3. **Predictive Modeling**
- Built using **Scikit-learn**.
- Features include Adult Mortality, GDP per Capita, BMI, and Years of Schooling.
- Displays metrics like R², RMSE, and MAE.
- Real-time prediction based on user input.

---

## 🚀 Quick Start

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/life-expectancy-analysis.git
   cd life-expectancy-analysis
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run life_expectancy.py
   ```

### Required Files
Ensure the following dataset is available in the project directory:
- `Life-Expectancy-Data-Averaged.csv`

---

## 📂 Repository Structure

```plaintext
.
├── life_expectancy.py       # Streamlit app script
├── requirements.txt         # Required Python packages
├── Life-Expectancy-Data-Averaged.csv  # Dataset
├── README.md                # Documentation
```

---

## ⚙️ Key Dependencies

- [Streamlit](https://streamlit.io/): Interactive app framework.
- [Pandas](https://pandas.pydata.org/): Data manipulation.
- [Plotly](https://plotly.com/python/): Advanced visualizations.
- [Scikit-learn](https://scikit-learn.org/): Machine learning modeling.
- [Seaborn](https://seaborn.pydata.org/): Statistical data visualization.

---

## 📊 Data Insights
- The dataset covers multiple countries and regions.
- Key columns include:
  - **Life Expectancy** (years)
  - **GDP per Capita**
  - **Schooling** (years)
  - **Adult Mortality** (per 1000)
  - **BMI** (Body Mass Index)
  - **Alcohol Consumption** (liters per capita)

---

## 🛠️ Development
### Predictive Model Workflow:
1. **Data Preprocessing:**
   - Handle missing values.
   - Normalize using `StandardScaler`.

2. **Feature Selection:**
   - Selected based on correlation analysis.

3. **Train-Test Split:**
   - Split dataset into training (80%) and testing (20%).

4. **Model Training:**
   - Linear Regression model to predict life expectancy.

### Performance Metrics:
- **R² Score:** Indicates model accuracy.
- **RMSE:** Average prediction error.
- **MAE:** Mean absolute error.

---

## 🎨 Future Enhancements
- Add support for additional datasets.
- Incorporate advanced models like Random Forest or XGBoost.
- Introduce more interactive visualizations.
- Provide country-specific recommendations.

---

## 👩‍💻 Contributing
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit changes and push:
   ```bash
   git push origin feature-branch
   ```
4. Open a pull request.

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 💡 Acknowledgments
- Dataset sourced from the [WHO Global Health Observatory](https://www.who.int/data/gho).
- Visualization techniques inspired by the **Plotly** and **Seaborn** libraries.
