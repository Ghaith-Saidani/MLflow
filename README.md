# End-to-End MLOps Pipeline with MLflow

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.17%2B-orange)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This repository provides a **complete, production-ready MLOps template** using **MLflow** as the central platform. It demonstrates an end-to-end machine learning workflow — from experiment tracking to model registry, serving, monitoring, logging, and interactive inference — all containerized and orchestrated with Docker Compose. 

[Image of MLOps pipeline components]


The example use case is a **Customer Churn Prediction** model trained on the classic `Churn_Modelling.csv` dataset.

---

## Features

* **MLflow Tracking & Model Registry** – Full experiment tracking, comparison, and model versioning
* **Model Serving** – REST API (`app.py`) serving the latest registered model
* **Interactive Demo** – Beautiful **Streamlit** app for real-time predictions
* **Observability Stack**
    * **Prometheus** + Node Exporter → System & application metrics
    * **Elasticsearch** → Centralized logging (via custom Python logger)
* **One-command Setup** – Powered by `Makefile` and `docker-compose.yml`
* **Automated Training Pipeline** – Reproducible runs with logged parameters, metrics, and artifacts
* **CI/CD Ready** – GitHub Actions workflow included

---

## Tech Stack

| Category | Tools / Libraries |
| :--- | :--- |
| **MLOps** | MLflow (Tracking + Registry + Serving) |
| **Modeling** | Scikit-learn, Pandas, NumPy |
| **API & Serving** | FastAPI / Flask (via `app.py`) |
| **Frontend** | Streamlit |
| **Containerization** | Docker + Docker Compose |
| **Monitoring** | Prometheus, Node Exporter |
| **Logging** | Elasticsearch (optional integration) |
| **Database** | SQLite (MLflow backend) – easily swappable to PostgreSQL |
| **Automation** | Makefile |

---

## Quick Start (Recommended)

### Prerequisites

* Docker & Docker Compose (v2+)
* GNU Make (optional but recommended)
* Git

### Step-by-Step

```bash
# 1. Clone the repository
git clone [https://github.com/Ghaith-Saidani/MLflow.git](https://github.com/Ghaith-Saidani/MLflow.git)
cd MLflow

# 2. Start the full stack (MLflow + Prometheus + App)
make up
# or: docker compose up -d --build

# 3. Train the model and register it in MLflow
make run
# This executes model_pipeline.py and logs everything to MLflow
```
4. Access the Services
```bash
ServiceURLPurpose
MLflow Tracking UIhttp://localhost:5000View experiments, compare runs, manage modelsModel REST APIhttp://localhost:8000/invocationsProduction inference endpointStreamlit Demo Apphttp://localhost:8501Interactive prediction dashboardPrometheushttp://localhost:9090Metrics dashboard
```
5. Test the Deployed Model
```Bash
# Using the provided test script
python test_request.py

# Or via curl (example)
curl -X POST http://localhost:8000/invocations \
     -H "Content-Type: application/json" \
     -d @data.json
```
Project Structure.
```bash
├── .github/workflows/          # GitHub Actions CI/CD
├── app.py                      # Model serving API (loads latest registered model)
├── main.py                     # Entry point (calls model_pipeline.py)
├── model_pipeline.py           # Full training pipeline + MLflow logging
├── streamlit_app.py            # Interactive Streamlit dashboard
├── test_request.py             # Simple API testing script
├── churn_modelling.csv         # Dataset
├── data.json                   # Sample input for inference
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Multi-stage build for the app
├── docker-compose.yml          # All services (MLflow, Prometheus, app, etc.)
├── Makefile                    # Convenience commands
├── mlflow.db                   # SQLite backend (created on first run)
├── prometheus/                 # Prometheus config & binaries
├── node_exporter/              # System metrics exporter
├── elasticsearch_logger.py     # Optional Elasticsearch logging integration
└── tests/                      # Unit & integration tests
```
TeardownStop and remove everything cleanly:
```Bash
make down
# or: docker compose down -v --remove-orphans
```
Next Steps & Improvements

Replace SQLite with PostgreSQL + S3/MinIO for production storage.

Add model drift detection (e.g., Evidently AI integration).

Deploy to Kubernetes (Helm chart ready in /k8s).

Add authentication to MLflow UI and API.

Integrate GitHub Actions for automated retraining.

Contributing

Contributions are very welcome! Feel free to:

Open issues for bugs or feature requests.

Submit pull requests (new models, monitoring improvements, etc.).

Improve documentation or add new examples.

Please follow the standard GitHub Flow.

License

This project is licensed under the MIT License – see the LICENSE file for details.

Made with ❤️ by Ghaith Saidani

⭐ Star this repo if it helped you level up your MLOps game!
