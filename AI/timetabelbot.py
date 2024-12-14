import pandas as pd
import schedule
import time
from flask import Flask, request, jsonify
import threading

# Load the CSV file
data = pd.read_csv(r'C:\Users\hp\OneDrive\Documents\AI\timetable.csv')

app = Flask(__name__)
current_question = None

# Function to ask a question
def ask_question(question):
    global current_question
    current_question = question
    print(f"Question: {question}")

# Schedule the questions based on the interval column in the CSV file
def schedule_questions():
    for index, row in data.iterrows():
        schedule.every(row['interval']).minutes.do(ask_question, question=row['question'])

# Run the scheduler in a separate thread
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

# Start the scheduler in a new thread
threading.Thread(target=run_scheduler).start()

# Flask route to get the current question
@app.route('/ask', methods=['GET'])
def ask():
    return jsonify({"question": current_question})

# Flask route to handle the response
@app.route('/respond', methods=['POST'])
def respond():
    response = request.json.get('response')
    # Process or store the response
    return jsonify({"status": "success"})

if __name__ == '__main__':
    schedule_questions()
    app.run(debug=True)
