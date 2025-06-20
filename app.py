from flask import Flask, render_template, request
import os
import re
import nltk
import pickle
import pandas as pd
from googleapiclient.discovery import build

nltk.download('punkt')

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

# Extract video ID from URL
def get_video_id(url):
    if "v=" in url:
        video_id = url.split("v=")[1]
        video_id = video_id.split("&")[0]
        video_id = video_id.split("?")[0]
        return video_id
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1]
        video_id = video_id.split("?")[0]
        return video_id
    return None

# Fetch YouTube comments using API
def get_comments(video_id):
    api_key = "AIzaSyBrBorsphnQjkX5wMT19mDsTAxFsiPU3gc"  # ← Replace this with your actual API key
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=50,
        textFormat="plainText"
    )
    response = request.execute()
    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)
    return comments

# Clean comment text
def clean_text(text):
    return re.sub(r"[^a-zA-Z ]", "", text).lower()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])

def analyze():
    url = request.form['url']
    video_id = get_video_id(url)
    if not video_id:
        return "<h3>Invalid YouTube URL</h3><a href='/'>Go back</a>"

    comments = get_comments(video_id)
    cleaned = [clean_text(c) for c in comments]
    vectors = vectorizer.transform(cleaned)
    predictions = model.predict(vectors)

    df = pd.DataFrame({'comment': comments, 'sentiment': predictions})
    counts = df['sentiment'].value_counts()

    result = "<h3>Sentiment Results:</h3><ul>"
    for sentiment in ['positive', 'neutral', 'negative']:
        count = counts.get(sentiment, 0)
        result += f"<li>{sentiment.capitalize()}: {count}</li>"
    result += "</ul><br><a href='/'>Analyze Another</a>"
    return result

if __name__ == '__main__':
    app.run(debug=True)
