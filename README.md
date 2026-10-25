# 🚗 AutoCast AI

## Statistical Time-Series Forecasting & AI-Powered Automotive Sales Decision Intelligence

> **AutoCast AI** is an end-to-end statistical forecasting and sales analytics application built to transform historical automotive sales transactions into interpretable business insights, validated forecasts, and decision-support reports. It combines data quality auditing, exploratory data analysis, time-series diagnostics, classical statistical forecasting models, chronological model evaluation, and an optional Groq-powered AI Copilot in an interactive Streamlit dashboard.

---

## 🚀 Overview

Automotive sales data can contain changing sales patterns, irregular order activity, cancelled transactions, and limited historical observations. Reliable forecasting therefore requires more than fitting a model to the available data: it also requires data-quality checks, appropriate time aggregation, careful validation, and transparent communication of uncertainty.

AutoCast AI provides a structured workflow for exploring automotive sales and forecasting future performance. Users can inspect the dataset, review sales trends, diagnose the monthly time series, compare statistical models, generate forecasts with prediction intervals, and export reports.

![AutoCast - AI Demo](https://github.com/user-attachments/assets/f79a57bf-a6c4-45d3-94f2-201cd4a5b2a1)

The application emphasizes **statistical validity and transparent model selection**. It uses chronological validation rather than random train-test splitting, keeps a final hold-out period untouched during model selection, compares candidate models against simple benchmarks, and reports forecast errors using clearly defined metrics.

### 🎯 Project Objectives

- Build an end-to-end statistical time-series forecasting workflow for automotive sales.
- Audit and clean transaction data while preserving the original raw dataset.
- Explore revenue, units, orders, customers, countries, and product-line performance.
- Diagnose trend, stationarity, and seasonal evidence before selecting model candidates.
- Compare ARIMA, SARIMA, ARIMAX, and SARIMAX using chronological validation.
- Evaluate models using out-of-sample error metrics and simple forecasting benchmarks.
- Produce future forecasts with 95% prediction intervals.
- Integrate an optional LLM-powered Copilot that explains data-grounded analytical outputs.
- Provide downloadable analytical and forecasting reports through an accessible dashboard.

---

## ✨ Key Features

### 📂 1. Automotive Sales Dataset

The project uses the **Auto Sales Data** dataset published on Kaggle by `ddosad`.

- **Dataset source:** [Auto Sales Data — Kaggle](https://www.kaggle.com/datasets/ddosad/auto-sales-data)
- **Dataset size documented in the project:** 2,747 order lines
- **Date range:** January 2018 – May 2020
- **Geographic coverage:** 19 countries
- **Product coverage:** 7 product lines

The app can use the CSV placed in the expected raw-data directory or allow the user to upload the dataset through the interface if the default file is unavailable.

### 🧹 2. Data Quality Audit & Cleaning

Before statistical modelling, the application checks and prepares the transaction data.

The workflow includes:

- Inspecting dataset dimensions, columns, and data types.
- Parsing order dates and checking date coverage.
- Identifying missing or suspicious values.
- Reviewing transaction status and allowing selected statuses to be excluded.
- Recording cleaning operations in an audit log.
- Keeping the raw dataset unchanged and writing cleaned data separately.
- Removing direct contact fields such as phone numbers, addresses, and contact names from processed data.
![data-quality](https://github.com/user-attachments/assets/2bca52b0-294a-4f6a-84a1-eaa24e279d17)
The data-quality dashboard helps users understand the condition of the dataset before drawing conclusions from it.

### 📊 3. Exploratory Data Analysis (EDA)

The analytics dashboard summarizes sales activity using business-relevant measures, including:

- Total revenue
- Units sold
- Distinct orders
- Average order value
- Customer count
- Country coverage
- Revenue by product line
- Monthly revenue trends
- Data-supported observations about sales patterns
![Sales Analytics Demo](https://github.com/user-attachments/assets/446f2a74-38ee-48d7-ae22-52606cfd75cb)
Revenue, units, and order counts are treated as distinct measures rather than being mixed together.

### 🔬 4. Time-Series Diagnostics

The diagnostic workflow examines whether the selected monthly sales series is suitable for the candidate forecasting models.

It includes:

- Monthly time-series construction.
- Augmented Dickey–Fuller (ADF) stationarity testing.
- Seasonal-strength assessment.
- Residual and model diagnostic visualizations.
- Checks that guide whether seasonal model candidates should be considered.
- 
![ts analysis](https://github.com/user-attachments/assets/a7c9eafc-2cfa-4ec3-a30f-cc223b2ff308)

Seasonal modelling is handled cautiously because the dataset contains only about 2.4 years of observations. A seasonal period of 12 is considered only when the minimum-cycle and seasonal-strength criteria are met. Seasonal evidence is treated as tentative rather than conclusive.

### 🧮 5. Statistical Forecasting Models

AutoCast AI compares four classical statistical model families:

| Model | Description |
|---|---|
| **ARIMA** | Autoregressive Integrated Moving Average; models temporal dependence and differencing. |
| **SARIMA** | Seasonal ARIMA; extends ARIMA with seasonal terms. |
| **ARIMAX** | ARIMA with exogenous explanatory variables. |
| **SARIMAX** | Seasonal ARIMA with exogenous variables. |


![Model Lab Demo](https://github.com/user-attachments/assets/0a29887f-62c1-439c-a1a3-50949a29214c)

The forecasting target is configurable:

- **Revenue:** sum of `SALES` by month (default)
- **Units:** sum of quantity by month
- **Orders:** distinct orders by month

Cancelled orders are excluded by default, and the status exclusions can be changed in the dashboard settings.

The default revenue series is monthly because it provides 29 gap-free monthly observations in the documented dataset; daily aggregation has many empty periods, while weekly aggregation is noisier.

### 🧪 6. Chronological Model Evaluation

The model-selection workflow is designed to reduce information leakage.

- The final **3 months** are reserved as a hold-out test set by default.
- The earlier development period is evaluated using **expanding-window rolling-origin validation**.
- The default validation design uses **6 folds**, each forecasting a **3-month horizon**, where supported by the available data and settings.
- Candidate models that fail a validation fold are not eligible for selection.
- Naive and seasonal-naive forecasts are included as reference benchmarks.
- The default primary selection metric is **WAPE**; MAE and RMSE can also be selected.
- A parsimony rule prefers the simpler model when candidates are within 5% of the best primary score.
- AIC and BIC are displayed as fit diagnostics, not used alone to select the best forecasting model.

This setup makes the evaluation more representative of forecasting future periods than a random split would be.

### 📏 7. Forecast Accuracy Metrics

The Model Lab reports common forecasting evaluation measures:

| Metric | Purpose |
|---|---|
| **MAE** | Average absolute forecast error, in the target's units. |
| **RMSE** | Penalizes larger errors more strongly than MAE. |
| **WAPE** | Total absolute error divided by total absolute actual values. |
| **MAPE** | Percentage-based error where actual values meet the implementation's positivity requirement. |
| **95% interval coverage** | Share of observed values contained within the 95% prediction intervals. |
| **AIC / BIC** | Information criteria used as model-fit diagnostics. |

![Forcast Demo](https://github.com/user-attachments/assets/1070ae06-1451-4e66-8eef-ce47672ee42b)

The dashboard distinguishes validation performance from final hold-out performance. A strong in-sample fit is not treated as proof of future forecast accuracy.

### 🔮 8. Forecast Studio & Prediction Intervals

Once model comparison is complete, the Forecast Studio allows users to generate forecasts for a selected future horizon.

Features include:

- Forecasts from the selected statistical model.
- Historical-versus-forecast visualizations.
- 95% prediction intervals.
- Forecast summaries in the selected target units.
- Clear separation between historical observations and future estimates.

Prediction intervals communicate uncertainty and should be interpreted alongside the short historical span and validation results.

### 🤖 9. AI Copilot — Groq Integration

The optional AI Copilot provides a natural-language interface to help users interpret the dashboard's analytical results.

- Uses Groq as the LLM provider when configured.
- Grounds answers in deterministic Python tools and available aggregates rather than sending the full transaction-level dataset to the LLM.
- Supports a demo mode when no API key is configured.
- Keeps the application’s core data analysis and forecasting features available without an LLM key.

The app checks model availability at runtime. Model candidates configured by the project include:

- `llama-3.3-70b-versatile`
- `openai/gpt-oss-120b`
- `openai/gpt-oss-20b`
- `llama-3.1-8b-instant`

Model availability can change, so the configured model should be checked against the models enabled for the user's Groq account. If a selected model is unavailable, the application can report the issue and fall back to deterministic demo responses.

### 📄 10. Reports & Downloads

The application can generate a PDF sales forecasting report containing relevant sections such as:

- Dataset summary and headline metrics
- Data-quality findings and cleaning audit log
- Exploratory analysis observations
- Time-series diagnostic results
- Statistical model evaluation
- Forecast summaries and supporting information

The report is designed to help communicate the analytical workflow and results in a shareable format.

---

## 🖥️ Dashboard Pages

AutoCast AI contains eight interactive Streamlit pages.

| Page | Purpose |
|---|---|
| **1. Executive Overview** | Headline KPIs, monthly revenue trend, product-line performance, and forecast summary when available. |
| **2. Data Quality** | Dataset checks, suspicious values, cleaning details, and audit information. |
| **3. Sales Analytics** | Descriptive sales analysis and business breakdowns. |
| **4. Time-Series Diagnostics** | Stationarity, seasonality, and time-series diagnostic views. |
| **5. Model Lab** | ARIMA-family model comparison, rolling-origin validation, hold-out testing, residual diagnostics, and candidate leaderboard. |
| **6. Forecast Studio** | Future forecasts, horizon selection, and prediction intervals. |
| **7. AI Copilot** | Natural-language interpretation of data-grounded analytical outputs, with optional Groq integration. |
| **8. Reports & Downloads** | Generate and download analytical reports. |

---

## 🏗️ End-to-End Workflow

```text
                 Automotive Sales CSV
                          │
                          ▼
                Data Loading & Validation
                          │
                          ▼
                Data Quality Audit/Cleaning
                          │
                          ▼
                 Exploratory Data Analysis
                          │
                          ▼
             Monthly Target-Series Construction
                          │
                          ▼
             Time-Series / Seasonal Diagnostics
                          │
                          ▼
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
     ARIMA              SARIMA             ARIMAX / SARIMAX
       └──────────────────┼──────────────────┘
                          ▼
             Expanding-Window Validation
                          │
                          ▼
           Model Selection + Naive Benchmarks
                          │
                          ▼
                 Final Hold-Out Evaluation
                          │
                          ▼
             Forecasts + 95% Prediction Intervals
                          │
             ┌────────────┴─────────────┐
             ▼                          ▼
       Streamlit Dashboard        PDF Report
             │
             ▼
    Optional Groq AI Copilot
    (aggregate-grounded answers)
```

---

## 🧠 Statistical Methodology & Key Design Decisions

### Forecast target and aggregation

The default target is monthly revenue, calculated as the sum of `SALES`. The dashboard also supports units and distinct orders as alternative targets. The chosen target and transaction-status exclusions are applied consistently across analysis and forecasting.

### Exogenous variables

The documented model design uses calendar flags for **October and November**, because future calendar values are known at forecast time. Variables such as quantity, price, order counts, and `days_since_last_order` are not used as future regressors because their future values are not known in advance. Unknown regressors are not silently replaced with zeros.

### Seasonality

The series is short, so yearly seasonality requires caution. A seasonal period of 12 is allowed only when there are at least two cycles and the seasonal-strength threshold is met. The observed evidence is considered fragile because it is driven by only two November peaks. Seasonal differencing is not used by default because it would consume a large share of the available observations.

### Model selection

Model selection uses chronological rolling-origin validation and a configurable error metric. The final hold-out period remains untouched until after model selection. The workflow also reports simple naive benchmarks and uses a parsimony rule to avoid selecting a more complex model for a negligible improvement.

### Uncertainty and interpretation

Forecasts include 95% prediction intervals. Because the dataset spans only about 2.4 years, the results should be interpreted as exploratory decision support rather than a guarantee of future sales. Descriptive associations do not establish causation.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Core application and statistical workflow |
| **Pandas** | Data cleaning, transformation, and aggregation |
| **NumPy** | Numerical operations |
| **Statsmodels** | ARIMA-family forecasting and statistical diagnostics |
| **Scikit-learn** | Supporting evaluation and analytical utilities |
| **Plotly** | Interactive visualizations |
| **Streamlit** | Multi-page interactive dashboard |
| **Groq SDK** | Optional LLM-powered AI Copilot |
| **python-dotenv** | Local environment-variable configuration |
| **ReportLab** | PDF report generation |
| **Pytest** | Automated testing |

---

## 📁 Project Structure

The project is organized into a Streamlit entry point, dashboard pages, reusable analytical modules, tests, and data/report directories.

```text
AutoCast-AI/
├── app.py
├── pages/
│   ├── 1_Executive_Overview.py
│   ├── 2_Data_Quality.py
│   ├── 3_Sales_Analytics.py
│   ├── 4_Time_Series_Diagnostics.py
│   ├── 5_Model_Lab.py
│   ├── 6_Forecast_Studio.py
│   ├── 7_AI_Copilot.py
│   └── 8_Reports_Downloads.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── diagnostics.py
│   ├── evaluation.py
│   ├── forecasting.py
│   ├── charts.py
│   ├── chatbot.py
│   ├── reporting.py
│   └── ui.py
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_diagnostics.py
│   ├── test_evaluation.py
│   ├── test_forecasting.py
│   └── test_chatbot.py
├── data/
│   ├── raw/
│   └── processed/
├── reports/
├── Auto_Sales_data.csv
├── cleaning_audit_log.json
├── requirements.txt
├── .env.example
└── README.md
```

> **Note:** This is the intended logical structure. Before publishing, make sure the directory names and files in the GitHub repository match the actual files you upload. The supplied project also includes a notebook and log file; include them if they are part of the version you want to publish.

---

## ⚙️ Installation

### 1. Clone the repository

Replace the placeholder below with your actual GitHub repository URL.

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd AutoCast-AI
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Place `Auto_Sales_data.csv` in `data/raw/` if you are using the default file path. If the default file is not present, use the app's dataset-upload option where available.

The raw dataset should remain unchanged; processed output is written separately.

---

## 🔐 Environment Variables — Optional Groq API

The AI Copilot can use Groq for natural-language explanations. The statistical dashboard and deterministic analysis workflow can run without an API key.

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Add your configuration:

```env
GROQ_API_KEY=your_groq_api_key_here
# Optional: set a model supported by your Groq account
GROQ_MODEL=llama-3.3-70b-versatile
```

Keep the actual key private. **Never commit `.env`, API keys, or credentials to GitHub.** Use `.env.example` to document variable names without real secret values.

---

## ▶️ Running the Application

Start the Streamlit dashboard from the project root:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

To run the automated test suite:

```bash
python -m pytest -q
```

The model comparison can take around 30 seconds the first time it runs, after which results are cached for the current settings and data file. Changing the global forecast settings can trigger a new model comparison; changing page-level display filters does not retrain the models.

---

## 💡 Example Use Cases

### 📈 Sales performance

- Which product line contributes the most revenue?
- How has monthly revenue changed over the available historical period?
- How many units and distinct orders were recorded?
- How do revenue, order count, and average order value differ?

### 🔬 Statistical analysis

- Does the monthly revenue series appear stationary under the ADF test?
- Is there enough evidence to consider yearly seasonality?
- How do ARIMA, SARIMA, ARIMAX, and SARIMAX compare under chronological validation?
- Does the selected model outperform naive benchmarks?
- How well do the 95% prediction intervals cover observed values?

### 🔮 Forecasting

- What is the forecast for the next selected number of months?
- How wide are the forecast prediction intervals?
- How does the selected model perform on the untouched hold-out period?
- How sensitive are model results to the target and validation settings?

### 🤖 AI Copilot

Users can ask for plain-language explanations of available analytical outputs, such as a summary of observed trends or an interpretation of model comparison results. Copilot responses are intended to support analysis, not replace statistical judgement.

---

## 🌟 Why AutoCast AI?

### Traditional sales reporting

```text
Transaction Data
      ↓
Manual Cleaning
      ↓
Descriptive Dashboard
      ↓
Manual Forecasting
      ↓
Separate Report Preparation
```

### AutoCast AI workflow

```text
Transaction Data
      ↓
Quality Audit + EDA
      ↓
Statistical Diagnostics
      ↓
Chronological Model Validation
      ↓
Forecasts + Uncertainty Intervals
      ↓
Interactive Dashboard + Reports
      ↓
Optional AI-Assisted Interpretation
```

AutoCast AI brings data preparation, statistical forecasting, evaluation, visualization, and reporting into one workflow. Its key emphasis is not just producing a forecast, but documenting how the forecast was selected and evaluated.

---

## ⚠️ Limitations

- **Short history:** The dataset covers approximately 2.4 years, limiting the strength of long-term seasonal conclusions.
- **Small hold-out:** The final test set contains only a few months, so hold-out metrics can vary substantially.
- **Fragile seasonality:** Yearly seasonal evidence is based on limited repeated cycles and should not be treated as conclusive.
- **Limited regressors:** Only future-known calendar flags are used as exogenous regressors in the documented design.
- **Prediction uncertainty:** Prediction intervals may not achieve nominal 95% coverage in such a short series.
- **Forecast shape:** Outside the calendar effects represented by the model, forecasts may be relatively flat.
- **Association is not causation:** Descriptive patterns do not establish why sales changed.
- **Dataset scope:** Results reflect this dataset and do not automatically generalize to all automotive businesses or markets.
- **AI dependency:** LLM-based prose requires a valid and available Groq model; deterministic analysis remains the source for the core numerical outputs.

---

## 🔮 Future Improvements

Possible extensions include:

- Incorporating longer and more recent automotive sales histories.
- Adding external drivers when reliable future values are available, such as planned promotions or macroeconomic indicators.
- Comparing the statistical models with additional forecasting approaches where the data volume supports them.
- Adding prediction-interval calibration and more extensive forecast backtesting.
- Supporting multi-product-line and multi-country forecasts with carefully designed validation.
- Adding scheduled forecast refreshes and monitoring for forecast error drift.
- Improving deployment, logging, and automated data-ingestion workflows.
- Expanding report customization and executive-summary exports.

These are potential future directions, not claims about functionality already implemented.

---

## 📚 Skills Demonstrated

This project demonstrates applied work in:

- Statistical Analysis
- Time-Series Analysis
- ARIMA, SARIMA, ARIMAX, and SARIMAX
- Stationarity Testing and Seasonal Diagnostics
- Chronological Cross-Validation
- Forecast Evaluation and Error Metrics
- Prediction Intervals and Uncertainty Communication
- Data Cleaning and Exploratory Data Analysis
- Python and Pandas
- Interactive Data Visualization
- Streamlit Application Development
- LLM API Integration
- Automated Testing
- PDF Report Generation

---

## 🏁 Project Goal

The goal of AutoCast AI is to provide a transparent, statistically grounded workflow for automotive sales forecasting—one that connects historical data, model diagnostics, out-of-sample evaluation, uncertainty-aware forecasts, and practical business interpretation.

> **From Automotive Sales Data to Statistically Validated Forecasts and AI-Assisted Decisions.**
