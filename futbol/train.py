import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# Load the dataset
data = pd.read_csv("large_mock_football_data.csv")

# Encode categorical features
encoder = LabelEncoder()
data["team_a_name"] = encoder.fit_transform(data["team_a_name"])
data["team_b_name"] = encoder.fit_transform(data["team_b_name"])
data["match_location"] = encoder.fit_transform(data["match_location"])
data["weather_condition"] = encoder.fit_transform(data["weather_condition"])


# Encode the target (result)
data["result"] = data["result"].map({"team_a_win": 0, "team_b_win": 1, "draw": 2})


# Split features and labels
X = data.drop("result", axis=1)
y = data["result"]


# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train

# Build the model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')  # 3 classes (team_a_win, team_b_win, draw)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=64, batch_size=2, validation_split=0.3)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the model
model.save("football_model.h5")



# cargar el modelo
model = tf.keras.models.load_model("football_model.h5")

# Predict for a new match
new_match = np.array([[8, 1, 8, 9, 1, 1, 1, 1]])  # Example data
new_match = scaler.transform(new_match)
prediction = model.predict(new_match)
predicted_class = np.argmax(prediction)
result_mapping = {0: "team_a_win", 1: "team_b_win", 2: "draw"}
print(f"Predicted Result: {result_mapping[predicted_class]}")
