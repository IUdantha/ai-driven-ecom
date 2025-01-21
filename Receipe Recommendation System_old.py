# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 16:53:14 2025

@author: Azmarah Rizvi
"""
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

def fetch_poster(recipe_name, recipe_id):
    # Construct the URL for the recipe
    url = f"https://www.food.com/recipe/{recipe_name}-{recipe_id}"
    
    try:
        # Fetch the webpage
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch URL: {url}")
            return None  # Return None if the webpage couldn't be fetched
        
        # Parse the webpage
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Locate the div with the class containing the image
        div_tag = soup.find('div', {'class': 'primary-image svelte-wgcq7z'})
        if div_tag:
            # Locate the img tag within the div
            img_tag = div_tag.find('img')
            if img_tag and 'srcset' in img_tag.attrs:
                # Extract the first URL from the srcset attribute
                srcset = img_tag['srcset']
                image_url = srcset.split(' ')[0]  # Get the first URL in the srcset
                return image_url
            elif img_tag and 'src' in img_tag.attrs:
                # Fallback to src attribute if srcset is not available
                return img_tag['src']
    
    except Exception as e:
        print(f"Error fetching poster for URL {url}: {e}")
    
    return None  # Return None if no image is found


# Load data and models
@st.cache_data
def load_recipes():
    # Load preprocessed recipes dataset
    recipes = pd.read_csv('C:\\Users\\Azmarah Rizvi\\Desktop\\ai-driven-ecom\\preprocessed_recipes.csv')
    return recipes

@st.cache_resource
def load_vectorizer():
    # Load TF-IDF vectorizer
    with open('C:\\Users\\Azmarah Rizvi\\Desktop\\ai-driven-ecom\\vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

@st.cache_resource
def load_tfidf_matrix():
    # Load TF-IDF matrix
    with open('C:\\Users\\Azmarah Rizvi\\Desktop\\ai-driven-ecom\\tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    return tfidf_matrix

# Main function to load all resources
def load_models_and_data():
    recipes = load_recipes()
    vectorizer = load_vectorizer()
    tfidf_matrix = load_tfidf_matrix()
    return recipes, vectorizer, tfidf_matrix

# Filter recipes that match all preferences
def filter_recipes_by_preferences(preferences, recipes):
    filtered_recipes = recipes.copy()
    
    # Check for each preference in 'tags_cleaned' column
    for pref in preferences:
        filtered_recipes = filtered_recipes[filtered_recipes['tags_cleaned'].str.contains(pref, case=False, na=False)]
    
    return filtered_recipes


# Recommend recipes based on user preferences
def recommend_recipes(preferences, recipes, vectorizer, tfidf_matrix):
    # Filter recipes based on preferences
    filtered_recipes = filter_recipes_by_preferences(preferences, recipes)

    if filtered_recipes.empty:
        return pd.DataFrame()  # Return empty DataFrame if no matches found

    # Get the indices of the filtered recipes in the original dataset
    filtered_indices = filtered_recipes.index

    # Extract the corresponding rows from the preloaded tfidf_matrix
    filtered_tfidf_matrix = tfidf_matrix[filtered_indices]

    # Combine user preferences into a single string query
    user_query = " ".join(preferences)

    # Transform the user query using the vectorizer
    user_vector = vectorizer.transform([user_query])

    # Compute cosine similarity between the user vector and the filtered TF-IDF matrix
    similarity_scores = cosine_similarity(user_vector, filtered_tfidf_matrix).flatten()

    # Rank recipes based on similarity scores
    # top_indices = similarity_scores.argsort()[-5:][::-1]  # Top 5 matches

    # Retrieve the top recommendations
    #recommended_recipes = filtered_recipes.iloc[top_indices]
    
    # Rank recipes based on similarity scores (all matches, highest to lowest)
    sorted_indices = similarity_scores.argsort()[::-1]  # All matches sorted in descending order

    # Retrieve all recommended recipes in sorted order
    recommended_recipes = filtered_recipes.iloc[sorted_indices]
    
    return recommended_recipes

# Streamlit app
def main():
    st.title("Recipe Recommendation System")
    st.write("Select your preferences, and we'll recommend recipes tailored to you!")

    # Load data and models
    recipes, vectorizer, tfidf_matrix = load_models_and_data()

    # User preferences
    st.sidebar.header("Your Preferences")
    diet = st.sidebar.selectbox("Dietary Preferences", ["Any", "Vegetarian", "Vegan", "Gluten-Free", "Keto", "Paleo"])
    cuisine = st.sidebar.selectbox("Cuisine", ["Any", "Italian", "Mexican", "Indian", "Chinese", "Mediterranean"])
    taste = st.sidebar.multiselect("Taste Preferences", ["Spicy", "Sweet", "Savory", "Sour"])
    ingredients = st.sidebar.text_input("Ingredients (comma-separated, optional):")

    # Combine preferences into a query
    preferences = []
    if diet != "Any":
        preferences.append(diet)
    if cuisine != "Any":
        preferences.append(cuisine)
    preferences.extend(taste)
    if ingredients:
        preferences.append(ingredients)

    # Generate recommendations
    if st.sidebar.button("Recommend Recipes"):
        if preferences:
            recommendations = recommend_recipes(preferences, recipes, vectorizer, tfidf_matrix)
            
            st.write("### Recommended Recipes:")
            for _, row in recommendations.iterrows():
                recipe_name = row['name'].replace(" ", "-").lower()
                recipe_id = row['id']
                poster_url = fetch_poster(recipe_name, recipe_id)
                st.write(f"#### **{row['name'].upper()}**")
                if poster_url:
                    st.image(poster_url, caption=row['name'], use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/500", caption="Image not available", use_container_width=True)

                st.write(f"*Description:* {row['tags']}")
                st.write(f"*Ingredients:* {', '.join(eval(row['ingredients']))}")
                st.write(f"*Steps:* {', '.join(eval(row['steps']))}")
                st.write("---")
        else:
            st.write("Please select at least one preference!")

if __name__ == "__main__":
    main()


