from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download tokenizer
nltk.download('punkt')

app = Flask(__name__)

# Load FAQ data
data = pd.read_csv("faq.csv")

questions = data['question'].tolist()
answers = data['answer'].tolist()

# Convert text into vectors
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# Function to get chatbot response
def get_response(user_input):
    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, question_vectors)

    best_match_index = similarity.argmax()

    best_score = similarity[0, best_match_index]

    if best_score < 0.2:
        return "Sorry, I don't understand the question."

    return answers[best_match_index]

# Home page
@app.route("/", methods=["GET", "POST"])
def home():
    response = ""

    if request.method == "POST":
        user_input = request.form["user_input"]
        response = get_response(user_input)

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)