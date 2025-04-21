import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import re
import pickle
import numpy as np
import joblib
import nltk
import requests
import json
import os
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load environment variables
load_dotenv()

# Download NLTK resources if not already available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Setup Porter Stemmer and stopwords
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')  # Keep negation words for sentiment analysis

class DeepSeekAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            print("Warning: DeepSeek API key not found in .env file")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        
    def analyze_sentiment(self, review):
        """Analyze sentiment using DeepSeek API"""
        if not self.api_key:
            raise ValueError("DeepSeek API key not found. Please check your .env file.")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Analyze the sentiment of this restaurant review and respond with only "positive" or "negative". 
        Review: "{review}"
        """
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1  # Low temperature for more consistent results
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            sentiment = result['choices'][0]['message']['content'].strip().lower()
            
            # Convert to binary prediction
            return 1 if sentiment == "positive" else 0
            
        except Exception as e:
            raise Exception(f"DeepSeek API error: {str(e)}")

class SentimentAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Restaurant Review Sentiment Analyzer")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize both traditional and DeepSeek analyzers
        try:
            print("Loading models...")
            self.cv = pickle.load(open("c1_BoW_Sentiment_Model.pkl", "rb"))
            self.classifier = joblib.load('c2_Classifier_Sentiment_Model')
            self.model_loaded = True
            print("Traditional models loaded successfully")
            
            self.deepseek = DeepSeekAnalyzer()
            print("DeepSeek analyzer initialized")
            
        except Exception as e:
            self.model_loaded = False
            print(f"Error loading models: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
        
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="Restaurant Review Sentiment Analyzer", 
                               font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#333333")
        title_label.pack(pady=20)
        
        # Instructions
        instructions = tk.Label(self.root, text="Enter a restaurant review to analyze its sentiment:",
                                font=("Arial", 12), bg="#f0f0f0", fg="#333333")
        instructions.pack(anchor="w", padx=20, pady=10)
        
        # Review input area
        self.review_input = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, 
                                                      width=70, height=8, 
                                                      font=("Arial", 12))
        self.review_input.pack(padx=20, pady=10, fill=tk.X)
        
        # Model selection frame
        model_frame = tk.Frame(self.root, bg="#f0f0f0")
        model_frame.pack(fill=tk.X, padx=20)
        
        # Model selection
        self.model_var = tk.StringVar(value="both")
        tk.Label(model_frame, text="Analysis Model:", bg="#f0f0f0", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        tk.Radiobutton(model_frame, text="Traditional", variable=self.model_var, 
                      value="traditional", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(model_frame, text="DeepSeek", variable=self.model_var, 
                      value="deepseek", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(model_frame, text="Both", variable=self.model_var, 
                      value="both", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        
        # Sample reviews frame
        samples_frame = tk.Frame(self.root, bg="#f0f0f0")
        samples_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Sample review buttons
        tk.Label(samples_frame, text="Try sample reviews:", bg="#f0f0f0", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        positive_sample = tk.Button(samples_frame, text="Positive Sample", 
                                   command=lambda: self.load_sample("The food was excellent and the service was amazing!"),
                                   font=("Arial", 10))
        positive_sample.pack(side=tk.LEFT, padx=5)
        
        negative_sample = tk.Button(samples_frame, text="Negative Sample",
                                   command=lambda: self.load_sample("The food was terrible and the service was horrible."),
                                   font=("Arial", 10))
        negative_sample.pack(side=tk.LEFT, padx=5)
        
        # Analyze button
        analyze_button = tk.Button(self.root, text="Analyze Sentiment", 
                                   command=self.analyze_sentiment,
                                   font=("Arial", 12, "bold"),
                                   bg="#4CAF50", fg="white",
                                   activebackground="#45a049",
                                   padx=20, pady=10)
        analyze_button.pack(pady=20)
        
        # Results frame
        self.results_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.results_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Traditional model results
        self.traditional_frame = tk.LabelFrame(self.results_frame, text="Traditional Model", bg="#f0f0f0", font=("Arial", 10, "bold"))
        self.traditional_frame.pack(fill=tk.X, pady=5)
        
        self.traditional_label = tk.Label(self.traditional_frame, text="", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.traditional_label.pack(pady=5)
        
        # DeepSeek results
        self.deepseek_frame = tk.LabelFrame(self.results_frame, text="DeepSeek Model", bg="#f0f0f0", font=("Arial", 10, "bold"))
        self.deepseek_frame.pack(fill=tk.X, pady=5)
        
        self.deepseek_label = tk.Label(self.deepseek_frame, text="", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.deepseek_label.pack(pady=5)
        
        # Debug info
        self.debug_label = tk.Label(self.results_frame, text="", 
                                    font=("Arial", 10),
                                    bg="#f0f0f0", wraplength=700)
        self.debug_label.pack(pady=10)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_sample(self, sample_text):
        """Load a sample review into the text input"""
        self.review_input.delete("1.0", tk.END)
        self.review_input.insert("1.0", sample_text)
        self.analyze_sentiment()

    def preprocess_review(self, review):
        # Clean and preprocess the review
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        return review
        
    def analyze_traditional(self, review):
        """Analyze using traditional ML model"""
        processed_review = self.preprocess_review(review)
        review_vector = self.cv.transform([processed_review]).toarray()
        prediction = self.classifier.predict(review_vector)[0]
        proba = self.classifier.predict_proba(review_vector)[0]
        return prediction, proba
    
    def analyze_sentiment(self):
        if not self.model_loaded and self.model_var.get() != "deepseek":
            messagebox.showerror("Error", "Traditional models not loaded correctly!")
            return
            
        review = self.review_input.get("1.0", tk.END).strip()
        
        if not review:
            messagebox.showwarning("Warning", "Please enter a review")
            return
            
        self.status_bar.config(text="Analyzing sentiment...")
        self.root.update_idletasks()
        
        try:
            model_choice = self.model_var.get()
            
            # Traditional model analysis
            if model_choice in ["traditional", "both"]:
                prediction, proba = self.analyze_traditional(review)
                confidence = proba[1] if prediction == 1 else proba[0]
                sentiment = "POSITIVE 😊" if prediction == 1 else "NEGATIVE 😞"
                color = "green" if prediction == 1 else "red"
                self.traditional_label.config(
                    text=f"{sentiment} (Confidence: {confidence:.2%})",
                    fg=color
                )
            else:
                self.traditional_label.config(text="Not used")
            
            # DeepSeek analysis
            if model_choice in ["deepseek", "both"]:
                try:
                    deepseek_prediction = self.deepseek.analyze_sentiment(review)
                    sentiment = "POSITIVE 😊" if deepseek_prediction == 1 else "NEGATIVE 😞"
                    color = "green" if deepseek_prediction == 1 else "red"
                    self.deepseek_label.config(
                        text=f"{sentiment}",
                        fg=color
                    )
                except Exception as e:
                    self.deepseek_label.config(
                        text=f"Error: {str(e)}",
                        fg="red"
                    )
            else:
                self.deepseek_label.config(text="Not used")
            
            self.status_bar.config(text="Analysis complete")
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_bar.config(text="Analysis failed")

# Run the application
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SentimentAnalyzerApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}") 