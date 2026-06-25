"""
Assignment 3: NLP Web-based Data Application
Milestone II: Web-based Data Application
Student Name: Anushka Sauran
Student ID: s4109010

Introduction:
This is an application-based project that was created using Flask for Assignment 3-Milestone II. 
Now, users can browse through a catalog of clothing, search using keywords, read detailed product descriptions, and leave reviews. 
The product design focuses on providing users with an intuitive and fun experience through machine learning-enhanced interactions with 
products and customer reviews.

The core feature of the application works through predictive analysis to find out if a customer is inclined to recommend a product based 
on their review text. To complete this work, the prior assignment generated a machine learning pipeline that combines TF-IDF weighting 
and FastText word embeddings with Logistic Regression classification. The system produces a binary recommendation label after users 
submit their reviews which indicates 1 for recommended items and 0 for not recommended items.

The platform enables users to browse products and submit reviews and it provides a keyword search tool which can identify both singular 
and plural versions of words like "dress" and "dresses". Every item page presents exhaustive review details together with suggested and 
definitive recommendation labels. This project demonstrates the application of natural language processing techniques in a real-world 
web application to simplify review analysis and improve user interaction.

"""
# Loading the necessary libraries
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from gensim.models import FastText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import numpy as np  
import os
from datetime import datetime

# Flask Application for Online Clothing Store with ML-based Review Recommendations
# This application allows users to browse clothing items, search for products, and create reviews with AI-powered recommendation predictions

# Initializing the Flask application with secret key for management
clothing_store_app = Flask(__name__)
clothing_store_app.secret_key = 'browsing-is-fun'

# Global variables to store the data and machine learning models 
# These will be loaded when the application starts
fasttext_word_model = None           # FastText model for the word embeddings
recommendation_classifier = None     # Logistic regression model for the predictions
text_frequency_vectorizer = None     # TF-IDF vectorizer for the text preprocessing
clothing_catalog_data = None         # DataFrame containing all the clothing items
customer_reviews_list = []           # List to store all the reviews given by customers

# Loading the pre-trained machine learning models and text vectorizer
# These models were created in the previous assignment for recommendation prediction

def load_machine_learning_models():
    global fasttext_word_model, recommendation_classifier, text_frequency_vectorizer
    try:
        # Loading the FastText Model
        fasttext_files = ["fasttext_model.bin"]
        fasttext_loaded = False
        for filename in fasttext_files:
            if os.path.exists(filename):
                try:
                    fasttext_word_model = FastText.load(filename)
                    print(f"FastText model loaded from: {filename}")
                    fasttext_loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")
                    continue
        if not fasttext_loaded:
            print("FastText model not found, using fallback prediction")
        
        # Loading the Logistic Regression Model
        if os.path.exists("logistic_regression_model.pkl"):
            with open("logistic_regression_model.pkl", "rb") as classifier_file:
                recommendation_classifier = pickle.load(classifier_file)
            print("Logistic regression model loaded successfully!")
        else:
            print("Logistic regression model not found")
        
        # Loading the TF-IDF Vectorizer
        if os.path.exists("tfidf_vectorizer.pkl"):
            with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
                text_frequency_vectorizer = pickle.load(vectorizer_file)
            print("TF-IDF vectorizer loaded successfully!")
        else:
            print("TF-IDF vectorizer not found")
        
        return fasttext_loaded and recommendation_classifier is not None

    except Exception as general_error:
        print(f"Error loading models: {general_error}")
        print("The application will continue with limited functionality")
        return False
    
# Loading the clothing catalog data from the CSV file
# This contains all the clothing items with their details and descriptions

def load_clothing_catalog():
    global clothing_catalog_data
    try:
        filename = "assignment3_II.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            required_columns = ['Clothing ID', 'Clothes Title', 'Clothes Description',
                                'Department Name', 'Class Name', 'Age', 'Rating']
            if not all(col in df.columns for col in required_columns):
                print(f"Missing one or more required columns.")
                print(f"Available columns: {df.columns.tolist()}")
                clothing_catalog_data = create_sample_clothing_data()
            else:
                clothing_catalog_data = df
                print(f"Loaded {len(df)} clothing items from {filename} successfully")
        else:
            print("File not found. Loading fallback.")
            clothing_catalog_data = create_sample_clothing_data()
    except Exception as e:
        print(f"Error loading catalog: {e}")
        clothing_catalog_data = create_sample_clothing_data()

# Creating a sample clothing data for testing when the main data file is not available
# This helps developers test the application without the full dataset

def create_sample_clothing_data():
    sample_clothing_items = {
        'Clothing ID': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],

        'Clothes Title': ['Elegant Summer Dress', 'Classic Blue Jeans', 'Sophisticated Evening Gown', 
            'Comfortable Cotton T-Shirt', 'Warm Winter Coat', 'Casual Denim Jacket', 
            'Professional Blazer', 'Cozy Knit Sweater'],

        'Clothes Description': [
            'Beautiful floral print summer dress perfect for warm weather occasions and casual outings',
            'High-quality denim jeans with comfortable fit, suitable for everyday wear and casual events',
            'Luxurious black evening gown designed for special occasions, parties, and formal events',
            'Soft premium cotton t-shirt available in multiple colors, perfect for casual everyday wear',
            'Insulated winter coat with waterproof exterior, designed to keep you warm in cold weather',
            'Classic denim jacket with vintage styling, perfect for layering and casual fashion',
            'Professional business blazer suitable for office wear and formal business meetings',
            'Comfortable knit sweater made from soft materials, ideal for cooler weather'
        ],

        'Department Name': ['Dresses', 'Bottoms', 'Dresses', 'Tops', 'Outerwear', 'Outerwear', 'Blazers', 'Sweaters'],
        'Class Name': ['Casual Dresses', 'Denim', 'Formal Dresses', 'Basic Tees', 'Winter Coats', 'Denim Jackets', 'Business Wear', 'Knitwear'],
        'Age': [35, 28, 42, 25, 38, 32, 45, 29],
        'Rating': [5, 4, 5, 3, 4, 4, 5, 4]
    }
    print("Created a sample clothing catalog with 8 test items")
    return pd.DataFrame(sample_clothing_items)

# Converting text into a numerical vector using FastText embeddings and TF-IDF weighting
# This creates a mathematical representation of the text that only the ML model can understand

def convert_text_to_vector(review_text, word_embedding_model, frequency_vectorizer):
    # To handle invalid or empty input
    if not review_text or not word_embedding_model or not frequency_vectorizer:
        return np.zeros(200)  
    
    # Spliting the text into individual words
    individual_words = review_text.split()

    try:
        # Calculating the TF-IDF weights for each word 
        tfidf_word_weights = frequency_vectorizer.transform([review_text]).toarray()[0]
        available_features = frequency_vectorizer.get_feature_names_out()

        # For creating weighted word vectors
        weighted_word_vectors = []

        # Processing each word in the review text
        for single_word in individual_words:
            if single_word in available_features:
                try:
                    # Finding the position and getting the word's embedding and multiplying by its TF-IDF weight 
                    word_position = available_features.tolist().index(single_word)
                    word_vector = word_embedding_model.wv[single_word] * tfidf_word_weights[word_position]
                    weighted_word_vectors.append(word_vector)
                except KeyError:
                    continue
        
        # Calculating the average of all word vectors to represent the entire text
        if len(weighted_word_vectors) > 0:
            return sum(weighted_word_vectors) / len(weighted_word_vectors)
        else:
            return np.zeros(word_embedding_model.vector_size)
        
    except Exception as vectorization_error:
        print(f"Error during text vectorization: {vectorization_error}")
        return np.zeros(200)
    
# Using the trained machine learning model to predict whether a customer would recommend an item
# And returns both the prediction (0 or 1) and the confidence level

def generate_recommendation_prediction(customer_review_text):
    try:
        if fasttext_word_model and recommendation_classifier and text_frequency_vectorizer:
            # Converting the review text to a numerical vector
            review_vector = convert_text_to_vector(
                customer_review_text, 
                fasttext_word_model, 
                text_frequency_vectorizer
            )

            # Making predictions using the trained classifier
            prediction_result = recommendation_classifier.predict([review_vector])[0]
            confidence_scores = recommendation_classifier.predict_proba([review_vector])[0]
            highest_confidence = confidence_scores.max()
            return int(prediction_result), float(highest_confidence)
        else:
            return create_fallback_prediction(customer_review_text)
            
    except Exception as prediction_error:
        print(f"Error in making recommendation prediction: {prediction_error}")
        return create_fallback_prediction(customer_review_text)
    
# A simple backup prediction method when ML models are not available
# This uses basic keyword analysis to determine recommendation
def create_fallback_prediction(review_text):
    # Lists of all the positive and negative sentiment words
    positive_sentiment_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'beautiful',
        'wonderful', 'fantastic', 'awesome', 'outstanding', 'brilliant', 'superb']
    
    negative_sentiment_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing',
        'poor', 'useless', 'waste', 'regret', 'uncomfortable', 'cheap']
    
    review_lowercase = review_text.lower() # Converting to lowercase 
    
    # Counting the positive and negative words
    positive_word_count = sum(1 for word in positive_sentiment_words if word in review_lowercase)
    negative_word_count = sum(1 for word in negative_sentiment_words if word in review_lowercase)
    
    # Making prediction based on sentiment analysis
    if positive_word_count > negative_word_count:
        return 1, 0.7  # Recommend 
    else:
        return 0, 0.6  # Don't recommend 
    
# This searches for the clothing items based on the user's keyword input
# Also this supports flexible matching including plural/singular forms

def search_clothing_items(search_keyword):
    if clothing_catalog_data is None or search_keyword is None:
        return [] 
    
    # Cleaning and preparing the search keyword
    cleaned_keyword = search_keyword.lower().strip()
    if not cleaned_keyword:
        return clothing_catalog_data.to_dict('records')
    
    # For handling plural/singular form variations
    keyword_variations = [cleaned_keyword]
    
    # Creating singular and plural forms
    if cleaned_keyword.endswith('es'):
        singular_form = cleaned_keyword[:-2]
        keyword_variations.append(singular_form)
    elif cleaned_keyword.endswith('s') and not cleaned_keyword.endswith('ss'):
        singular_form = cleaned_keyword[:-1]
        keyword_variations.append(singular_form)
    else:
        keyword_variations.append(cleaned_keyword + 's')
    
    # Adding some common clothing terminology
    clothing_synonyms = {
        'jean': ['jeans', 'denim', 'pants'],
        'dress': ['dresses', 'gown', 'frock'],
        'shirt': ['shirts', 'top', 'blouse', 'tee'],
        'pant': ['pants', 'trouser', 'trousers', 'jeans'],
        'coat': ['coats', 'jacket', 'outerwear'],
        'sweater': ['sweaters', 'jumper', 'pullover', 'knitwear']
    }
    
    # Expand keyword variations with synonyms
    for variation in keyword_variations.copy():
        if variation in clothing_synonyms:
            keyword_variations.extend(clothing_synonyms[variation])
    
    # Remove duplicates and creating search filter for matching items
    unique_keywords = list(dict.fromkeys(keyword_variations))
    search_filter = pd.Series([False] * len(clothing_catalog_data))
    
    # This will search across multiple columns for each keyword variation
    for keyword_variant in unique_keywords:
        search_filter |= (
            clothing_catalog_data['Clothes Title'].str.lower().str.contains(keyword_variant, na=False) |
            clothing_catalog_data['Clothes Description'].str.lower().str.contains(keyword_variant, na=False) |
            clothing_catalog_data['Department Name'].str.lower().str.contains(keyword_variant, na=False) |
            clothing_catalog_data['Class Name'].str.lower().str.contains(keyword_variant, na=False)
        )
    
    # Getting the matching items 
    matching_items = clothing_catalog_data[search_filter].to_dict('records')
    
    # Adding a relevance scoring system 
    for item in matching_items:
        relevance_score = 0
        item_title_lower = item['Clothes Title'].lower()
        
        # Items with keyword in title get higher relevance
        for keyword_variant in unique_keywords:
            if keyword_variant in item_title_lower:
                relevance_score += 2
            elif keyword_variant in item['Clothes Description'].lower():
                relevance_score += 1
        item['search_relevance'] = relevance_score
    
    # Sortting results by highest first
    matching_items.sort(key=lambda item: item.get('search_relevance', 0), reverse=True)
    return matching_items


# Flask Routes

# This will display the main homepage with the available clothing items
# This is the first page users see when they visit the store
@clothing_store_app.route('/')
def display_homepage():
    # Getting all the clothing items. 
    all_clothing_items = clothing_catalog_data.to_dict('records') if clothing_catalog_data is not None else []
    featured_items_limit = 20  # Number of items displaying on the home page
    
    # Selecting only the first few 'featured_items_limit' items
    featured_items = all_clothing_items[:featured_items_limit] if len(all_clothing_items) > featured_items_limit else all_clothing_items
    print(f"DEBUG: Sending {len(featured_items)} featured items to homepage (out of {len(all_clothing_items)} total)")
    return render_template('home.html', 
                           items=featured_items, 
                           total_catalog_items=len(all_clothing_items)) 

# This handles the user's search requests and display matching clothing items
# Users can also search by keywords like 'jeans', 'coat', 'shirt', etc.
@clothing_store_app.route('/search')
def handle_search_request():
    # Getting the search keyword from URL parameters
    user_search_keyword = request.args.get('keyword', '')
    
    # Finding the matching clothing items and displaying it ti the user
    search_results = search_clothing_items(user_search_keyword)
    return render_template('search_results.html', 
                         keyword=user_search_keyword, 
                         results=search_results, 
                         count=len(search_results))

# This will display detailed information about a specific clothing item
# Including the item details and the customer reviews
@clothing_store_app.route('/item/<int:clothing_item_id>')
def show_item_details(clothing_item_id):
    # Checking if clothing data is available or not
    if clothing_catalog_data is None:
        flash('Clothing catalog is currently unavailable. Please try again later.', 'error')
        return redirect(url_for('display_homepage'))
    
    # Finding that specific clothing item
    matching_items = clothing_catalog_data[clothing_catalog_data['Clothing ID'] == clothing_item_id].to_dict('records')
    if not matching_items:
        flash('The requested clothing item could not be found.', 'error')
        return redirect(url_for('display_homepage'))
    selected_item = matching_items[0]
    
    # Getting all the reviews for that specific item
    item_specific_reviews = [
        review for review in customer_reviews_list 
        if review['clothing_id'] == clothing_item_id 
    ]
    return render_template('item_detail.html', 
                         item=selected_item, 
                         reviews=item_specific_reviews)

# This will allow the customer to create new reviews for clothing items
# Including AI-powered recommendation prediction

@clothing_store_app.route('/create_review/<int:clothing_item_id>', methods=['GET', 'POST'])
def create_customer_review(clothing_item_id):
    if clothing_catalog_data is None:
        flash('Clothing catalog is currently unavailable.', 'error')
        return redirect(url_for('display_homepage'))
    target_item = clothing_catalog_data[clothing_catalog_data['Clothing ID'] == clothing_item_id].to_dict('records')
    
    if not target_item:
        flash('Cannot create review: clothing item not found.', 'error')
        return redirect(url_for('display_homepage'))
    selected_clothing_item = target_item[0]
    
    # To handle the POST request
    if request.method == 'POST':
        # Extracting the data from user input
        review_title = request.form.get('title', '').strip()
        review_description = request.form.get('review_text', '').strip()
        rating_str = request.form.get('rating', '').strip()
        customer_rating = 3  # Default rating 

        # Validating the rating input
        if rating_str and rating_str.isdigit():
            try:
                temp_rating = int(rating_str)
                if temp_rating in range(1, 6):
                    customer_rating = temp_rating
                else:
                    flash("Rating must be between 1 and 5. Defaulting to 3 stars.", "error")
            except ValueError:
                flash("Invalid rating format. Defaulting to 3 stars.", "error")
        else:
            flash("Rating is missing or invalid. Defaulting to 3 stars.", "error")
        
        # Combining the title and description for analysis and generating AI recommendation prediction
        complete_review_text = f"{review_title} {review_description}"
        ai_prediction, prediction_confidence = generate_recommendation_prediction(complete_review_text)
        
        # Checking if the user wants to override the AI prediction or not
        user_override_choice = request.form.get('recommendation')
        if user_override_choice is not None and user_override_choice != '': 
            final_recommendation = int(user_override_choice)
        else:
            final_recommendation = ai_prediction
        
        # This will create a new review record
        new_customer_review = {
            'id': len(customer_reviews_list) + 1,
            'clothing_id': clothing_item_id,
            'title': review_title,
            'review_text': review_description,
            'rating': customer_rating,
            'recommended': final_recommendation,
            'predicted_recommendation': ai_prediction,
            'confidence': prediction_confidence,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Adding the review to the collection 
        # And letting the user know about their reviews
        customer_reviews_list.append(new_customer_review)
        flash('Thank you! Your review has been successfully created.', 'success')
        return redirect(url_for('show_item_details', clothing_item_id=clothing_item_id))
    return render_template('create_review.html', item=selected_clothing_item)

# This will be the API endpoint for real-time recommendation prediction
# Used by JavaScript to provide instant feedback to users

@clothing_store_app.route('/api/predict_recommendation', methods=['POST'])
def api_recommendation_prediction():
    # Getting the review text from the request
    request_data = request.json
    review_text_input = request_data.get('text', '')
    
    # Generating the prediction using the ML model and returning the prediction in JSON format
    prediction_result, confidence_level = generate_recommendation_prediction(review_text_input)
    return jsonify({
        'prediction': prediction_result,
        'confidence': confidence_level,
        'recommendation_text': 'Recommended' if prediction_result == 1 else 'Not Recommended'
    })

# This will display all the customer reviews across the clothing items
# This is useful for browsing what other customers are saying
@clothing_store_app.route('/reviews')
def display_all_reviews():
    return render_template('all_reviews.html', reviews=customer_reviews_list)

# For handling 404 errors
@clothing_store_app.errorhandler(404)
def handle_page_not_found(error):
    flash('The page you requested could not be found.', 'error')
    return redirect(url_for('display_homepage'))

# For handling the internal server errors
@clothing_store_app.errorhandler(500)
def handle_server_error(error):
    flash('An internal error occurred. Please try again later.', 'error')
    return redirect(url_for('display_homepage'))

# This will injwct global variables into the templates
# Makes common information available across all pages
@clothing_store_app.context_processor
def inject_template_variables():
    return {
        'store_name': 'ChicShack',
        'current_year': datetime.now().year,
        'total_items': len(clothing_catalog_data) if clothing_catalog_data is not None else 0
    }


# Application Start
if __name__ == '__main__':
    print("Starting the ChicShack Application...")
    print("-" * 70)

    # Loading the machine learning models and data
    models_loaded_successfully = load_machine_learning_models()
    load_clothing_catalog()
    print("-" * 70)

    if models_loaded_successfully:
        print("Application ready with full AI functionality!")
    else:
        print("Application running with limited functionality (no ML models)")

    print("Starting the web server...")
    print("Access the application at: http://localhost:5000")
    print("-" * 70)

    # Starting the Flask development server
    clothing_store_app.run(
        debug=True,          
        host='0.0.0.0',      # Accept connections from any IP
        use_reloader=False,  
        port=5000            # Run on port 5000
    )