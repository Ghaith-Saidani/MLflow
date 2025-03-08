from elasticsearch import Elasticsearch
import mlflow
import logging
from datetime import datetime

# Connect to Elasticsearch with authentication
es = Elasticsearch(
    [{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],
    basic_auth=('elastic', 'MyS3cureP@ssw0rd2025!')
)

# Check if Elasticsearch is reachable
if es.ping():
    print("Connected to Elasticsearch!")
else:
    print("Elasticsearch ping failed")

# Check cluster health
try:
    health = es.cluster.health()
    print("Cluster Health:", health)
except Exception as e:
    print("Error fetching cluster health:", e)


# Function to log MLflow events to Elasticsearch
def log_to_elasticsearch(log_message, log_level='INFO'):
    log = {
        "@timestamp": datetime.utcnow(),
        "log_level": log_level,
        "message": log_message
    }

    # Send the log to Elasticsearch
    try:
        es.index(index="mlflow-logs", body=log)
        print(f"Logged to Elasticsearch: {log_message}")
    except Exception as e:
        print(f"Error logging to Elasticsearch: {e}")


# Configure logging for MLflow
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlflow")
logger.setLevel(logging.INFO)

# Example of logging an event with MLflow
logger.info("MLflow logger is successfully configured.")

# Example usage: Log an MLflow metric and log a message
mlflow.log_metric('example_metric', 1.23)
log_to_elasticsearch("This is a test log message from MLflow.")
