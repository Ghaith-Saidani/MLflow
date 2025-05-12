import logging
from elasticsearch import Elasticsearch
from logging import Handler


class ElasticsearchHandler(Handler):
    def __init__(self, es_client, index_name="mlflow_logs"):
        self.es_client = es_client
        self.index_name = index_name
        super().__init__()

    def emit(self, record):
        log_entry = self.format(record)
        try:
            self.es_client.index(index=self.index_name, document={"log": log_entry})
        except Exception as e:
            print(f"Error sending log to Elasticsearch: {e}")
