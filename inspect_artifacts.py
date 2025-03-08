import joblib

# Load and inspect X_processed
X_processed = joblib.load("mlartifacts/X_processed.pkl")
print("X_processed shape:", X_processed.shape)
print("First 5 rows of X_processed:", X_processed[:5])

# Load and inspect y_processed
y_processed = joblib.load("mlartifacts/y_processed.pkl")
print("y_processed shape:", y_processed.shape)
print("First 5 labels of y_processed:", y_processed[:5])

# Load and inspect scaler
scaler = joblib.load("mlartifacts/scaler.pkl")
print("Scaler object:", scaler)

# Load and inspect PCA
pca = joblib.load("mlartifacts/pca.pkl")
print("PCA object:", pca)
