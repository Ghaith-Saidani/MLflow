# End-to-End MLOps Pipeline with MLflow

This repository provides a comprehensive template for implementing an **end-to-end MLOps pipeline** using **MLflow** for experiment tracking, model registration, and serving. The project trains a classification model (likely predicting customer churn from `churn_modelling.csv`), containers the entire infrastructure using Docker Compose, and includes monitoring and logging components.

## ✨ Features

* **Experiment Tracking:** Use **MLflow** to track model parameters, metrics, and artifacts (`.pkl` files).
* **Model Lifecycle Management:** Register, stage, and transition models using the MLflow Model Registry.
* **Containerization:** Full stack deployment using **Docker** and **Docker Compose**.
* **Model Serving:** Expose the trained model via a REST API using `app.py`.
* **Monitoring & Observability:** Integration with **Prometheus** for metrics collection and **Elasticsearch** for centralized logging.
* **Interactive Demo:** A **Streamlit** application (`streamlit_app.py`) for demonstrating model inference.
* **Automation:** Simple commands for environment setup and execution via a **Makefile**.

## 🛠️ Tech Stack

| Category | Tool / Library |
| :--- | :--- |
| **MLOps** | MLflow |
| **Data Science** | Python, Scikit-learn (implied) |
| **Containerization** | Docker, Docker Compose |
| **Monitoring** | Prometheus, Node Exporter |
| **Logging** | Elasticsearch |
| **API** | Flask/FastAPI (implied by `app.py`) |
| **Frontend** | Streamlit |
| **Database** | SQLite (`mlflow.db`) |

## 🚀 Getting Started

### Prerequisites

You need the following installed on your machine:

* **Docker**
* **Docker Compose** (or Docker with Compose V2)
* `make` (optional, for simplified commands)

### 1. Setup and Initialization

Use the `Makefile` to build the Docker image and start all required services (MLflow server, Prometheus, etc.) defined in `docker-compose.yml`.

```bash
make up
This command performs the following actions:Builds the project Docker image.Starts the services: MLflow Tracking Server, Prometheus, Node Exporter, and the main application (app.py).2. Run the ML Training PipelineExecute the main training script to run the model pipeline (model_pipeline.py) and log the experiment to the MLflow Tracking Server.Bashmake run
This will:Preprocess the churn_modelling.csv data.Train a model.Log parameters and metrics (e.g., accuracy, loss).Save model artifacts (model.pkl, scaler.pkl, pca.pkl).Register the model in the MLflow Model Registry.3. Access UIsOnce the services are running, you can access the various UIs:ServiceURLPurposeMLflow Tracking UIhttp://localhost:5000View experiments, compare runs, manage models.Prometheus UIhttp://localhost:9090Query and visualize system metrics.Streamlit Apphttp://localhost:8501Interactive interface for model inference.4. Model Inference (Testing the API)You can test the deployed model's API endpoint locally.Run the request test script:Bashpython test_request.py
Alternatively, if the API is exposed on a specific port (e.g., 8000 or 5001 based on typical app.py setups):Bashcurl -X POST http://localhost:<PORT>/predict -H "Content-Type: application/json" -d @data.json
📂 Project Structure.
├── .github/workflows/
├── app.py                  # API for model serving/inference
├── main.py                 # Entry point for running MLflow experiment
├── model_pipeline.py       # Data preprocessing and model training logic
├── streamlit_app.py        # Streamlit dashboard for model demo
├── churn_modelling.csv     # Dataset used for training
├── requirements.txt        # Python dependencies
├── Dockerfile              # Defines the application container image
├── docker-compose.yml      # Orchestrates MLflow, Prometheus, and application
├── Makefile                # Automation commands (up, run, clean)
├── db.sqlite               # Database for the MLflow tracking backend
├── prometheus-*/           # Prometheus binaries and configuration
├── node_exporter-*/        # Node Exporter binaries for host metrics
├── elasticsearch_*.py      # Scripts for Elasticsearch logging integration
└── tests/                  # Unit and integration tests
🛑 Teardown
To stop and remove all running Docker containers and volumes:
make down
