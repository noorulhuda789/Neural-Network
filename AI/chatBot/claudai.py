import numpy as np
import tensorflow as tf
import nltk
import pandas as pd
import random
import string
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load and preprocess the data
data = pd.read_csv('chatbot_data.csv')  # You'll need to create this CSV file
questions = data['question'].tolist()
answers = data['answer'].tolist()

# Tokenize and lemmatize the text
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in string.punctuation]
    return tokens

# Create vocabulary
vocab = set()
for question in questions:
    vocab.update(preprocess_text(question))

vocab_size = len(vocab)
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for word, i in word_to_index.items()}

# Convert text to sequences
max_sequence_length = max(len(preprocess_text(question)) for question in questions)

def text_to_sequence(text):
    tokens = preprocess_text(text)
    sequence = [word_to_index.get(token, 0) for token in tokens]
    sequence = sequence + [0] * (max_sequence_length - len(sequence))
    return sequence

X = np.array([text_to_sequence(question) for question in questions])
y = np.array([answers.index(answer) for answer in answers])

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128, input_length=max_sequence_length),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(answers), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Function to get response
def get_response(text):
    sequence = text_to_sequence(text)
    sequence = np.array([sequence])
    prediction = model.predict(sequence)
    answer_index = np.argmax(prediction)
    return answers[answer_index]

# Chatbot loop
print("Chatbot: Hello! How can I assist you today? (Type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Chatbot: Goodbye!")
        break
    response = get_response(user_input)
    print("Chatbot:", response)