🧠 ANN Binary Classification with Streamlit & TensorFlow

This project implements a Binary Classification Artificial Neural Network (ANN) using TensorFlow/Keras for the backend and Streamlit for the user-facing frontend.

⸻

🎯 Project Goal

To allow users to interactively input data through a Streamlit web app and receive predictions based on a trained ANN model for a binary classification problem for customer churn in a bank

⸻

🏗️ Model Architecture

The ANN is constructed using a Sequential neural network consisting of:
	•	An input layer connected to the training features
	•	Two hidden layers using ReLU activation to learn complex patterns
	•	A single output layer using Sigmoid activation to predict binary outcomes (0 or 1)

⸻

⚙️ Training Setup

The model was trained using:
	•	Binary Crossentropy as the loss function (ideal for binary classification)
	•	The Adam optimizer with a custom learning rate
	•	Accuracy as the performance metric

To improve model training:
	•	EarlyStopping was used to halt training when validation loss stopped improving
	•	TensorBoard was configured for training performance visualization
	•	The model was trained and evaluated on training/testing splits, and the best weights were restored

⸻

🖥️ Streamlit Frontend

The frontend was built using Streamlit, allowing real-time interaction with the trained model

⸻

💾 Model Saving

The trained model is saved in .keras format and is automatically loaded by the Streamlit app for inference.

⸻

📦 Tech Stack
	•	TensorFlow / Keras – Model building, training, and saving
	•	Pandas / NumPy / Scikit-learn – Data preprocessing and transformation
	•	Streamlit – Frontend UI for model interaction
