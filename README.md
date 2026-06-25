🛍️ ChicShack — NLP-Powered Clothing Review Web App

Overview
A full-stack Flask web application for an online clothing store featuring AI-powered review recommendation prediction. Users can browse products, search by keyword, and submit reviews — with a trained NLP pipeline predicting whether a review is recommended or not in real-time.

Features
Keyword search with singular/plural matching and clothing synonym expansion
AI-powered recommendation prediction on customer reviews
Real-time confidence scoring via REST API endpoint
Fallback sentiment analysis when ML models are unavailable

How It Works
1. User submits a review for a clothing item
2. Review text is vectorized using TF-IDF weighting + FastText word embeddings
3. Logistic Regression model predicts: Recommended (1) or Not Recommended (0)
4. Confidence score is returned alongside the prediction

Tools & Technologies
1. Backend: Python, Flask
2. NLP & ML: FastText, TF-IDF, Logistic Regression, Scikit-learn
3. Data Processing: Pandas, NumPy
4. Frontend: HTML, CSS, Jinja2 templates

How to Run
1. Clone the repository
2. Install dependencies:
pip install flask gensim scikit-learn pandas numpy
3. Run the app:
python app.py
4. Open http://localhost:5000 in your browser

Data Source
Women's Clothing E-Commerce Reviews dataset

Note: The pre-trained model files (fasttext_model.bin, logistic_regression_model.pkl, tfidf_vectorizer.pkl) are not included due to file size limits. The app will automatically run in fallback mode using keyword-based sentiment analysis.
