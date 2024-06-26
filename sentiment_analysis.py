
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

# Load the IMDb dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)

# Pad sequences to ensure a consistent length
max_review_length = 500  # Or the length that best suits your model
X_train = pad_sequences(X_train, maxlen=max_review_length)
X_test = pad_sequences(X_test, maxlen=max_review_length)

# Create the model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(5000, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))  # Adjust LSTM units as needed
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

# Compile the model with the appropriate loss function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
print(model.summary())

# Fit the model with proper training data
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)  # Verbose for more feedback

# Evaluate the model on the test set
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1] * 100))
