from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

# Load data and models
def load_models_and_data():
    recipes = pd.read_csv('preprocessed_recipes.csv')
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    return recipes, vectorizer, tfidf_matrix

def fetch_poster(recipe_name, recipe_id):
    url = f"https://www.food.com/recipe/{recipe_name}-{recipe_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.content, 'html.parser')
        div_tag = soup.find('div', {'class': 'primary-image svelte-wgcq7z'})
        if div_tag:
            img_tag = div_tag.find('img')
            if img_tag and 'srcset' in img_tag.attrs:
                return img_tag['srcset'].split(' ')[0]
            elif img_tag and 'src' in img_tag.attrs:
                return img_tag['src']
    except Exception as e:
        print(f"Error fetching poster for recipe: {recipe_name}-{recipe_id}: {e}")
    return None

def filter_recipes_by_preferences(preferences, recipes):
    filtered_recipes = recipes.copy()
    for pref in preferences:
        filtered_recipes = filtered_recipes[filtered_recipes['tags_cleaned'].str.contains(pref, case=False, na=False)]
    return filtered_recipes

def recommend_recipes(preferences, recipes, vectorizer, tfidf_matrix):
    filtered_recipes = filter_recipes_by_preferences(preferences, recipes)
    if filtered_recipes.empty:
        return pd.DataFrame()
    filtered_recipes = filtered_recipes.reset_index(drop=True)
    filtered_indices = filtered_recipes.index
    filtered_tfidf_matrix = tfidf_matrix[filtered_indices]
    user_query = " ".join(preferences)
    user_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(user_vector, filtered_tfidf_matrix).flatten()
    sorted_indices = similarity_scores.argsort()[::-1]
    recommended_recipes = filtered_recipes.iloc[sorted_indices]
    return recommended_recipes

app = Flask(__name__)

recipes, vectorizer, tfidf_matrix = load_models_and_data()

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Recipe Recommendation API! Use the '/recommend' endpoint to get recommendations."})

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        preferences = []
        if data.get('preference'):
            preferences.append(data['preference'])
        if data.get('Cuisine'):
            preferences.append(data['Cuisine'])
        if data.get('taste'):
            preferences.extend(data['taste'].split(","))
        if data.get('ingredients'):
            preferences.append(data['ingredients'])

        recommendations = recommend_recipes(preferences, recipes, vectorizer, tfidf_matrix)
        if recommendations.empty:
            return jsonify({"message": "No recipes found matching your preferences."}), 404

        response = []
        for _, row in recommendations.iterrows():
            recipe_name = row['name'].replace(" ", "-").lower()
            recipe_id = row['id']
            poster_url = fetch_poster(recipe_name, recipe_id)
            response.append({
                "name": row['name'],
                "tags": row['tags'],
                "ingredients": eval(row['ingredients']),
                "steps": eval(row['steps']),
                "poster_url": poster_url
            })
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=4321)
