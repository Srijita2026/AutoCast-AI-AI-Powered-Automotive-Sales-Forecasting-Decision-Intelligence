# AutoCast AI

### AI-Powered Automotive Sales Forecasting & Decision Intelligence

AutoCast AI is an end-to-end analytics and forecasting application built
with **Python and Streamlit**. It helps users explore automotive sales
data, assess data quality, diagnose time-series behaviour, compare
statistical forecasting models, and turn results into decision-oriented
insights through an optional Groq-powered AI Copilot.

The application combines reproducible data preparation, statistical
diagnostics, chronological model validation, forecast uncertainty
intervals, interactive visualizations, and downloadable reports in one
dashboard.

------------------------------------------------------------------------

## Table of Contents

-   [Project Overview](#project-overview)
-   [Key Features](#key-features)
-   [Dashboard Pages](#dashboard-pages)
-   [Dataset](#dataset)
-   [Forecasting Methodology](#forecasting-methodology)
-   [Technology Stack](#technology-stack)
-   [Project Structure](#project-structure)
-   [Installation](#installation)
-   [Run the Application](#run-the-application)
-   [Configure the AI Copilot](#configure-the-ai-copilot)
-   [Run Tests](#run-tests)
-   [Data Quality and Privacy](#data-quality-and-privacy)
-   [Limitations](#limitations)
-   [Deployment](#deployment)

------------------------------------------------------------------------

## Project Overview

Automotive sales teams need to understand historical performance, detect
patterns, and estimate future demand or revenue to support planning.
AutoCast AI provides a single workflow for exploring sales data and
evaluating forecasts rather than relying on a model's fit statistics
alone.

By default, the application forecasts **monthly revenue**, calculated as
the sum of `SALES`, with cancelled orders excluded. Users can also
select units or distinct orders as the target. Forecasts include 95%
prediction intervals, and the dashboard explains the validation approach
and important caveats.

**Dataset source:** [Auto Sales Data --- Kaggle
(ddosad)](https://www.kaggle.com/datasets/ddosad/auto-sales-data)

The project documentation describes the dataset as containing 2,747
order lines from January 2018 to May 2020, across 19 countries and 7
product lines.

## Key Features

-   **Data quality audit:** inspect schema, missing values, duplicates,
    invalid records, suspicious values, and cleaning actions.
-   **Reproducible preprocessing:** normalize column names, parse dates,
    validate essential fields, and preserve an audit log.
-   **Exploratory sales analytics:** review KPIs, monthly trends,
    growth, product lines, countries, and unusual periods.
-   **Time-series diagnostics:** inspect frequency, trend, seasonal
    evidence, stationarity, ACF/PACF, decomposition, and residual
    behaviour.
-   **Statistical forecasting:** compare ARIMA, SARIMA, ARIMAX, and
    SARIMAX where the available data supports them.
-   **Chronological evaluation:** use expanding-window rolling-origin
    validation and a separate final hold-out period.
-   **Model comparison:** evaluate forecasts with WAPE, MAE, RMSE, MAPE
    when appropriate, bias, and prediction-interval coverage.
-   **Forecast Studio:** generate future forecasts with 95% prediction
    intervals and review model-selection rationale.
-   **AI Copilot:** ask questions about historical performance, model
    metrics, forecasts, and scenarios using deterministic Python tools,
    with optional Groq language-model explanations.
-   **Scenario analysis:** explore user-defined adjustments without
    confusing scenario assumptions with historical observations or model
    forecasts.
-   **Reports and downloads:** export analysis and forecast results,
    including a PDF report.
-   **Automated tests:** test data cleaning, diagnostics, evaluation
    metrics, forecasting logic, and Copilot tool behaviour.

## Dashboard Pages

The Streamlit application has eight pages:

  -----------------------------------------------------------------------
  Page                                Purpose
  ----------------------------------- -----------------------------------
  **Executive Overview**              High-level sales KPIs and business
                                      summary

  **Data Quality**                    Data checks, suspicious records,
                                      and cleaning audit

  **Sales Analytics**                 Sales trends, breakdowns, growth,
                                      and unusual periods

  **Time-Series Diagnostics**         Frequency, trend, seasonality,
                                      stationarity, and residual
                                      diagnostics

  **Model Lab**                       Candidate models, validation
                                      results, and model selection

  **Forecast Studio**                 Future forecasts, prediction
                                      intervals, and forecast
                                      interpretation

  **AI Copilot**                      Natural-language questions grounded
                                      in deterministic analysis tools

  **Reports & Downloads**             Downloadable results and reports
  -----------------------------------------------------------------------

## Dataset

Download the dataset from
[Kaggle](https://www.kaggle.com/datasets/ddosad/auto-sales-data) and
place the CSV at:

``` text
data/raw/Auto_Sales_data.csv
```

The app can also provide a data-upload workflow if the expected local
file is not available. Check the exact filename expected by your local
project configuration before running the app.

The raw dataset is not modified by the cleaning workflow. Processed data
and audit outputs are kept separately from the raw input.

## Forecasting Methodology

### Target and time frequency

-   Default target: monthly revenue, calculated as `sum(SALES)`.
-   Alternative targets: units (`sum(QUANTITYORDERED`) and distinct
    orders.
-   Cancelled orders are excluded by default; this setting is
    configurable.
-   Monthly aggregation is used because the dataset is sparse at daily
    frequency and weekly values are noisy.

### Candidate models

The model-selection workflow compares applicable candidates from:

-   **ARIMA** --- autoregressive integrated moving average.
-   **SARIMA** --- ARIMA with seasonal terms.
-   **ARIMAX** --- ARIMA with exogenous regressors.
-   **SARIMAX** --- seasonal ARIMA with exogenous regressors.

Seasonal models are considered only when there is enough history and the
seasonal evidence passes the configured gate. Naive and seasonal-naive
forecasts are also included as benchmarks.

### Validation and selection

The forecasting pipeline is designed to reduce temporal leakage:

1.  Reserve the final three observations as a hold-out set by default.
2.  Use expanding-window rolling-origin validation on the development
    data.
3.  Score candidate models with a configurable primary metric, WAPE by
    default.
4.  Mark candidates that fail validation folds as ineligible under the
    default settings.
5.  Apply a parsimony rule: when models are within the configured
    tolerance of the best score, prefer the simpler model.
6.  Evaluate the hold-out only after model selection.
7.  Refit the selected model on all available observations before
    generating future forecasts.

AIC and BIC are retained as diagnostics, but model selection does not
rely on AIC alone.

### Exogenous variables and prediction intervals

The implementation uses calendar indicators for October and November
because their future values are known. Variables such as future prices,
quantities, and order counts are not assumed to be known. The
forecasting code refuses unknown or incomplete future regressors rather
than silently filling them with zero.

Forecasts include nominal 95% prediction intervals. Their actual
coverage should be assessed using the validation and hold-out results; a
95% interval does not guarantee that 95% of future observations will
fall inside it.

## Technology Stack

-   **Language:** Python
-   **Application:** Streamlit
-   **Data processing:** pandas, NumPy
-   **Forecasting and statistics:** statsmodels, scikit-learn
-   **Visualization:** Plotly
-   **AI integration:** Groq API (optional)
-   **Environment configuration:** python-dotenv
-   **PDF reporting:** ReportLab
-   **Testing:** pytest

## Project Structure

The repository is organized around a Streamlit entry point, dashboard
pages, reusable application modules, and tests.

``` text
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
├── requirements.txt
├── .env.example
└── README.md
```

Some directories and generated files may be created by the application
at runtime.

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Replace the placeholder URL and folder name with your GitHub repository
details.

### 2. Create and activate a virtual environment

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the dataset

Place the downloaded CSV in `data/raw/` using the filename expected by
the application, or use the in-app upload option if available.

## Run the Application

From the project root, run:

``` bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually
`http://localhost:8501`. Open that URL in your browser.

## Configure the AI Copilot

The AI Copilot is optional. Without a Groq API key, the application can
run in **demo mode**, where deterministic tool results are available
without LLM-generated prose.

1.  Create a `.env` file in the project root, using `.env.example` as a
    template if it is included.
2.  Add your API key:

``` dotenv
GROQ_API_KEY=your_groq_api_key_here
```

3.  Optionally set a model if supported by your configuration:

``` dotenv
GROQ_MODEL=llama-3.3-70b-versatile
```

4.  Restart the Streamlit app.

**Security:** Never commit `.env` or expose your API key in source code,
screenshots, notebooks, logs, or public issues. Add `.env` to
`.gitignore` and use your hosting provider's
secrets/environment-variable settings when deploying. The Copilot is
designed to use approved deterministic tools and aggregated results
rather than sending the full raw dataset to the language model.

## Run Tests

Run the test suite from the project root:

``` bash
python -m pytest -q
```

The tests cover data cleaning and audit behaviour, time-series
diagnostics, metric calculations, chronological validation and
forecasting, and AI Copilot tools and fallback behaviour.

## Data Quality and Privacy

-   Cleaning operations are recorded in an audit log.
-   Exact duplicates and invalid essential records are handled
    explicitly.
-   Missing values are not blindly replaced with zero.
-   Unusual sales values are flagged for review rather than
    automatically deleted.
-   Contact-related fields such as phone numbers, addresses, and contact
    names are excluded from processed analytical data.
-   Historical facts, model forecasts, and scenario assumptions are
    labelled separately in Copilot tool results.

Before publishing the repository, review the dataset's licence and
redistribution terms. If you cannot redistribute the raw CSV, include
instructions for downloading it instead of committing the dataset.

## Limitations

-   The dataset covers a relatively short historical period, limiting
    the reliability of long-range and seasonal conclusions.
-   A small number of yearly cycles makes seasonal
    patterns---particularly peaks around November---fragile.
-   The final hold-out period is short and may not contain a seasonal
    peak, so it cannot fully validate peak-month forecasting.
-   Prediction intervals may under-cover compared with their nominal 95%
    level; review empirical coverage in the evaluation output.
-   Forecasts are based on historical patterns and known calendar
    indicators. They do not establish causal relationships or
    automatically account for future market shocks, promotions, supply
    constraints, or pricing changes.
-   Results are decision-support estimates, not guarantees of future
    sales.

## Deployment

The application can be deployed to Streamlit Community Cloud or another
Python hosting environment that supports Streamlit.

General deployment steps:

1.  Push the source code to GitHub.
2.  Configure the required Python dependencies from `requirements.txt`.
3.  Make the dataset available through the expected data path or the
    app's upload workflow.
4.  Add `GROQ_API_KEY` through the hosting platform's secrets settings
    if you want the LLM-enabled Copilot.
5.  Start the app with:

``` bash
streamlit run app.py
```

Check the host's storage and file-persistence rules before relying on
generated reports or processed data being retained between restarts.

------------------------------------------------------------------------

## Author

Developed as a data science project combining statistical time-series
forecasting, exploratory analytics, model evaluation, and AI-assisted
decision support.

If you find this project useful, feel free to star the repository or
share feedback through GitHub Issues.
