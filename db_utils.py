import sqlite3
from datetime import datetime

# Define the SQLite database file
DB_PATH = "db.sqlite"  # Use the path relative to the working directory of your app


def create_predictions_table():
    """Create a predictions table if it doesn't exist already."""
    # Connect to SQLite database (this will create the database file if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # SQL statement to create the predictions table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS prediction_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        prediction_date TEXT NOT NULL,
        input_data TEXT NOT NULL,
        predicted_value REAL NOT NULL
    );
    """

    # Execute the create table query
    cursor.execute(create_table_query)
    conn.commit()
    print("✅ Predictions table created (if it didn't exist).")

    # Close the connection
    cursor.close()
    conn.close()


def save_prediction(model_name, input_data, predicted_value):
    """Save the prediction result to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get the current date and time
    prediction_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # SQL query to insert the prediction result
    insert_query = """
    INSERT INTO prediction_results (model_name, prediction_date, input_data, predicted_value)
    VALUES (?, ?, ?, ?);
    """

    # Execute the insert query
    cursor.execute(
        insert_query, (model_name, prediction_date, input_data, predicted_value)
    )
    conn.commit()

    print(f"✅ Prediction result saved to database: {predicted_value}")

    # Close the connection
    cursor.close()
    conn.close()


if __name__ == "__main__":
    create_predictions_table()
