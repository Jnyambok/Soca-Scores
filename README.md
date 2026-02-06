# ⚽ Soca Scores - MLOps Pipeline




## First Article in the Series : [ Phase 0: Setting up the Development Environment](https://medium.com/p/1960fb160e30)
## Second Article in the Series :[ Phase 1: Data Ingestion](https://medium.com/data-ai-and-beyond/building-a-full-stack-mlops-system-predicting-the-2025-2026-english-premier-league-season-phase-c9c1d4f83187)
## Third Article in the Series :[ Phase 2: Data Cleaning and Transformation](https://medium.com/data-ai-and-beyond/building-a-full-stack-mlops-system-predicting-the-2025-2026-english-premier-league-season-phase-8760a79ddfe1)

[Data Project Structure Best Practices](https://medium.com/the-pythonworld/best-practices-for-structuring-a-python-project-like-a-pro-be6013821168)


> **A comprehensive machine learning operations (MLOps) project for predicting English Premier League match outcomes using historical data and advanced feature engineering.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![MLOps](https://img.shields.io/badge/MLOps-Enabled-green)](https://ml-ops.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---
To run python -m src. components.<module.py>
Please note that some workflows and/or tools might be added or deleted at my discretion as per my needs.


## 🎯 **Project Overview**
![alt text](image.png)<br />

This project implements a comprehensive MLOps pipeline for predicting Premier League match outcomes, including win/loss/draw probabilities, as well as goal predictions. Built with production-ready practices, the system ingests data from footballdata.uk, processes historical match data, engineers meaningful features, trains predictive models, and serves predictions through an interactive Streamlit dashboard.

### **🏆 Key Features**
- **Automated Data Ingestion** from footballdata.uk
- **Database Storage** through a serverless PostgreSQL DB
- **Feature Store** for managing engineered features
- **Model Registry** with versioning and experiment tracking
- **Real-time Predictions** via Streamlit dashboard
- **MLOps Best Practices** with CI/CD, monitoring, and automated retraining
- **Modular Architecture** allowing independent component execution

---

## 🚀 **Approach & Methodology**

### **🔄 My MLOps Lifecycle approach**
```
Data Ingestion → Data Cleaning,Transformation and DB Loading → EDA -> Feature Engineering and Feature Store Storage → Model Training → Evaluation → Deployment → Monitoring
        ↑                                                                                                                                                              ↓
        ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←← **Continuous Improvement Loop** ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

### **📊 Data Strategy**
- **Source**: Historical Premier League data from footballdata.UK (CSV format)
- **Scope**: Multiple seasons of match data with team statistics
- **Updates**: Weekly ingestion during active season
- **Quality**: Automated validation and cleaning pipelines

### **🤖 Machine Learning Approach**
- **Problem Type**: Multi-class classification (Win/Draw/Loss) + Regression (Goal prediction - optional or future consideration)
- **Feature Engineering**: Team form, head-to-head records, player statistics, seasonal trends
- **Model Selection**: Ensemble methods with hyperparameter optimization
- **Evaluation**: Cross-validation with time-series split for temporal data

### **🏗️ MLOps Architecture**
- **Orchestration**: Pipeline automation for data processing and model training
- **Feature Store**: Centralized feature management with versioning
- **Model Registry**: Automated model versioning and promotion
- **Monitoring**: Data drift detection and model performance tracking
- **Deployment**: Containerized applications with CI/CD integration

---

## 📁 **Expected Project Structure (Might Change as per needed requirements)** 

```
premier_league_predictions/  
│
├── 📊 data/                        # Data storage layers
│   ├── common_data/                        # Raw CSV files from footballdata.uk containing URLs and any other common usable data
│   ├── ingested_data/                      # Merged datasets
│   ├── cleaned_data/                       # Cleaned datasets for DB loading
│   └── feature_store_data/                 # Transformed feature stores
|   └── predictions/                        # Model predictions output
│
├── 🧪 experiments/                 # ML experimentation workspace
│   ├── notebooks/                  # Jupyter notebooks for EDA
│   ├── scripts/                    # Experimental scripts
│   └── results/                    # Experiment outputs and metrics
│
├── 🔧 src/                         # Core application modules
│   ├── data_ingestion/             # Data collection and validation
│   │   ├── data_ingestion.py       # ⭐ Main ingestion script
│   │   ├── data_validator.py       # Data quality checks
│   │   └── data_merger.py          # CSV merging logic
│   │
│   ├── feature_store/              # Feature engineering pipeline
│   │   ├── feature_engineering.py  # Feature creation
│   │   ├── feature_store.py        # Feature storage & retrieval
│   │   └── feature_validator.py    # Feature quality assurance
│   │
│   ├── models/                     # ML model management
│   │   ├── training.py             # ⭐ Model training script
│   │   ├── inference.py            # Prediction generation
│   │   ├── model_registry.py       # Model versioning
│   │   └── evaluation.py           # Performance evaluation
│   │
│   ├── monitoring/                 # MLOps monitoring
│   │   ├── data_drift.py           # Data drift detection
│   │   ├── model_drift.py          # Model performance monitoring
│   │   └── alerts.py               # Alerting system
│   │
│   └── utils/                      # Shared utilities
│       ├── config.py               # Configuration management
│       ├── logger.py               # Logging utilities
│       └── database.py             # Database connections
│
├── 🚀 deployment/                  # Application deployment
│   ├── streamlit_app/              # Interactive dashboard
│   ├── api/                        # REST API (optional)
│   └── docker/                     # Containerization
│
├── 🔄 pipelines/                   # Orchestration scripts
│   ├── data_pipeline.py            # End-to-end data processing
│   ├── training_pipeline.py        # Model training pipeline
│   └── inference_pipeline.py       # Prediction pipeline
│
├── ⚙️ configs/                     # Configuration files
│   ├── data_config.yaml            # Data ingestion parameters
│   ├── model_config.yaml           # ML model configurations
│   └── app_config.yaml             # Application settings
│
├── 🧪 tests/                       # Comprehensive testing
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── fixtures/                   # Test data
│
└── logger
└── logger

```

## 📁 **Current Project Structure** 
```

Directory structure:
└── jnyambok-soca-scores/
    ├── README.md
    ├── data_dictionary_spreadsheet.html  #Added a html page for the data_dictionary
    ├── requirements.txt
    ├── logs/
    ├── datasets/
    │   └── common_data/
    │       ├── english_league_data_urls.csv
    │       └── feature_catalog.csv
    ├── experiments/
    │   ├── notebooks/
    │   │   └── data_ingestion.ipynb
    │   └── scripts/             #Added a scripts environ to test scripts
    │       └── data_cleaning.py
    └── src/
        ├── __init__.py
        ├── exception.py
        ├── logger.py
        └── components/
            ├── __init__.py
            ├── data_cleaning.py       #Added a script for cleaning the data before the database
            ├── data_ingestion.py    
            └── database_scripts/
                └── db_creation_and_initial_insertion.py  #For the initial creation and insertion

```

---

## 🛠️ **Technology Stack**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Data Processing** | `pandas`, `numpy` | Data manipulation and analysis |
| **Machine Learning** | `scikit-learn` | Model training and evaluation |
| **Feature Store** | `Feast` / Custom | Feature management and serving |
| **Database** | `NEON DB` | Data persistence |
| **Orchestration** | `Apache Airflow`, `Prefect` | Pipeline automation |
| **Experiment Tracking** | `MLflow` | Model versioning and tracking |
| **Frontend** | `Streamlit` | Interactive dashboard |
| **API** | `FastAPI` | REST API services |
| **Containerization** | `Docker` | Application packaging |
| **CI/CD** | `GitHub Actions` | Automated deployment |
| **Monitoring** | `Grafana`, `Prometheus` | System and model monitoring |

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+
- PostgreSQL (optional, SQLite for local development)
- Docker (for containerized deployment)

### **Installation**
```bash
# Clone the repository
git clone https://github.com/yourusername/premier_league_predictions.git
cd premier_league_predictions

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Run Individual Components**
```bash
# Data ingestion
python -m src.data_ingestion.data_ingestion

# Feature engineering
python -m src.feature_store.feature_engineering

# Model training
python -m src.models.training

# Generate predictions
python -m src.models.inference

# Start Streamlit dashboard
streamlit run deployment/streamlit_app/app.py
```

### **Run Complete Pipeline**
```bash
# Execute full MLOps pipeline
python scripts/run_full_pipeline.py
```

---

## 📊 **Model Performance Goals**

### **Target Metrics**
- **Match Outcome Accuracy**: >55% (industry benchmark ~52%)
- **Goal Prediction MAE**: <1.2 goals per match
- **Confidence Calibration**: Well-calibrated probability predictions

### **Business Value**
- Provide insights for football analytics
- Support betting strategy research (educational purposes)
- Demonstrate MLOps best practices in sports analytics



## 📈 **Roadmap**

### **Phase 1: Data Ingestion** ✅ (Finished on 08/19)
- [x] Project structure and configuration
- [x] Loading dataset urls
- [x] Ingesting data from data sources (footballdata.uk)
- [x] Merging the various datasets and storing locally

### **Phase 2: Data Cleaning x Transformation and Database Loading**
- [ ] Feature store implementation
- [ ] Model registry setup
- [ ] Streamlit dashboard
- [ ] Basic monitoring

### **Phase 3: Advanced Features** 📋
- [ ] Advanced feature engineering (player data, weather)
- [ ] Model ensemble and optimization
- [ ] Real-time prediction API
- [ ] Comprehensive monitoring dashboard

### **Phase 4: Production** 🚀
- [ ] Cloud deployment
- [ ] Automated CI/CD
- [ ] Performance optimization
- [ ] User feedback integration

---

## 🤝 **Contributing**

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### **Development Workflow**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎯 **Acknowledgments**

- **footballdata.uk** for providing comprehensive historical match data
- **Premier League** for the exciting matches that make this project possible
- **Open Source Community** for the amazing tools and libraries

---


*Built with ❤️ for football analytics and MLOps excellence*
