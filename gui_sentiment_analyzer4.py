import tkinter as tk
from tkinter import ttk
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.special import softmax
import torch

# Initialize Tkinter
root = tk.Tk()
root.title("Sentiment Analysis GUI")
root.geometry("800x400")  # Set the initial window size

# Add border around the page
root.configure(borderwidth=5, relief="solid")

# Load Roberta model and tokenizer
MODEL =  "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Create a function to analyze sentiment using VADER
def analyze_vader_sentiment():
    review = entry.get()
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(review)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment_label = "Positive"
    elif compound_score <= -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    result_label.config(text=f"VADER Sentiment: {sentiment_label}\nCompound Score:{compound_score:.2f}")

# Create a function to analyze sentiment using Roberta
def analyze_roberta_sentiment():
    review = entry.get()
    encoded_text = tokenizer(review, return_tensors='pt')
    output = model(**encoded_text)
    scores = output.logits.softmax(dim=-1).detach().numpy()[0]
    labels = ['Negative', 'Neutral', 'Positive']
    roberta_sentiment = labels[scores.argmax()]
    result_label.config(text=f"Roberta Sentiment: {roberta_sentiment}\nNegative Score: {scores[0]:.2f}\nNeutral Score: {scores[1]:.2f}\nPositive Score: {scores[2]:.2f}")

# Create a function to compare VADER and Roberta sentiments
def compare_sentiments():
    review = entry.get()
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(review)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        vader_sentiment = "Positive"
    elif compound_score <= -0.05:
        vader_sentiment = "Negative"
    else:
        vader_sentiment = "Neutral"
    
    encoded_text = tokenizer(review, return_tensors='pt')
    output = model(**encoded_text)
    scores = output.logits.softmax(dim=-1).detach().numpy()[0]
    labels = ['Negative', 'Neutral', 'Positive']
    roberta_sentiment = labels[scores.argmax()]
    
    result_label.config(text=f"VADER Sentiment: {vader_sentiment}\nCompound Score: {compound_score:.2f}\nRoberta Sentiment: {roberta_sentiment}\nNegative Score: {scores[0]:.2f}\nNeutral Score: {scores[1]:.2f}\nPositive Score: {scores[2]:.2f}")

# Add introductory text
intro_text = tk.Label(root, text="Welcome to Sentiment Analysis!\nAnalyze text using VADER and Roberta models.")
intro_text.pack(pady=10)

# Label for review entry
label = ttk.Label(root, text="Enter a review:")
label.pack()

# Entry field
entry = ttk.Entry(root, width=60)
entry.pack(padx=20, pady=5)

# Styling Functions
def apply_button_styling(widget, color):
    style = ttk.Style()
    style.configure("TButton", background=color, foreground='white', padding=10)

# Buttons
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

vader_button = ttk.Button(button_frame, text="Analyze with VADER", command=analyze_vader_sentiment)
apply_button_styling(vader_button, 'green')
vader_button.pack(side="left", padx=10)

roberta_button = ttk.Button(button_frame, text="Analyze with Roberta", command=analyze_roberta_sentiment)
apply_button_styling(roberta_button, 'blue')
roberta_button.pack(side="left", padx=10)

compare_button = ttk.Button(button_frame, text="Compare Sentiments", command=compare_sentiments)
apply_button_styling(compare_button, 'purple')
compare_button.pack(side="left", padx=10)

# Styling Functions
def apply_label_styling(widget):
    widget.configure(font=("Helvetica", 14, "bold"))

# Frame for model info
model_info_frame = ttk.Frame(root)
model_info_frame.pack(padx=20, pady=10, fill="both", expand=True)

# Vader Info Box
vader_info = tk.Label(model_info_frame, text="VADER Model\n\nVADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that provides positive, negative, and neutral sentiment scores.\n\nCompound score ranges from -1 (negative) to +1 (positive).", bg="#eaffdb", padx=10, pady=10, borderwidth=2, relief="solid", wraplength=300, justify="left")
vader_info.pack(side="left", fill="both", expand=True)

# Roberta Info Box
roberta_info = tk.Label(model_info_frame, text="Roberta Model\n\nRoberta is a transformer-based neural network model pre-trained on large text corpora. It analyzes text in context to determine sentiment scores across negative, neutral, and positive categories.", bg="#dbedea", padx=10, pady=10, borderwidth=2, relief="solid", wraplength=300, justify="left")
roberta_info.pack(side="left", fill="both", expand=True)

# Label to display result
result_label = ttk.Label(root, text="")
result_label.pack(pady=10)


# Run the GUI event loop
root.mainloop()

