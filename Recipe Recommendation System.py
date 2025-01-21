# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import pickle
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

# Persistent storage file
SELECTED_RECIPES_FILE = "C:\\Users\\Azmarah Rizvi\\Desktop\\ai-driven-ecom\\selected_recipes.json"

# Ensure session state is initialized properly
if "view_selected" not in st.session_state:
    st.session_state.view_selected = False  # Default to showing recommendations

if "selected_recipes" not in st.session_state:
    st.session_state.selected_recipes = {}

# Load selected recipes from JSON at start
def load_selected_recipes():
    if os.path.exists(SELECTED_RECIPES_FILE):
        with open(SELECTED_RECIPES_FILE, "r") as f:
            return json.load(f)
    return {}

# Save selected recipes to JSON
def save_selected_recipes():
    with open(SELECTED_RECIPES_FILE, "w") as f:
        json.dump(st.session_state.selected_recipes, f, indent=4)

# Load previously selected recipes into session state
st.session_state.selected_recipes = load_selected_recipes()


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
    except:
        return None
    return None


@st.cache_data
def load_recipes():
    return pd.read_csv('C:\\Users\\Azmarah Rizvi\\Desktop\\ai-driven-ecom\\preprocessed_recipes.csv')


@st.cache_resource
def load_vectorizer():
    with open('C:\\Users\\Azmarah Rizvi\\Desktop\\ai-driven-ecom\\vectorizer.pkl', 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_tfidf_matrix():
    with open('C:\\Users\\Azmarah Rizvi\\Desktop\\ai-driven-ecom\\tfidf_matrix.pkl', 'rb') as f:
        return pickle.load(f)


def load_models_and_data():
    recipes = load_recipes()
    vectorizer = load_vectorizer()
    tfidf_matrix = load_tfidf_matrix()
    return recipes, vectorizer, tfidf_matrix


def recommend_recipes(preferences, included, excluded, additional_prefs, recipes, vectorizer, tfidf_matrix):
    filtered_recipes = recipes.copy()

    for pref in preferences:
        filtered_recipes = filtered_recipes[filtered_recipes['tags_cleaned'].str.contains(pref, case=False, na=False)]

    for ing in included:
        filtered_recipes = filtered_recipes[filtered_recipes['ingredients'].str.contains(ing, case=False, na=False)]

    for ing in excluded:
        filtered_recipes = filtered_recipes[~filtered_recipes['ingredients'].apply(lambda x: ing.lower() in str(x).lower())]

    if filtered_recipes.empty:
        return pd.DataFrame()

    filtered_indices = filtered_recipes.index
    filtered_tfidf_matrix = tfidf_matrix[filtered_indices]

    user_query = " ".join(preferences + included + [additional_prefs])
    user_vector = vectorizer.transform([user_query])

    similarity_scores = cosine_similarity(user_vector, filtered_tfidf_matrix).flatten()
    sorted_indices = similarity_scores.argsort()[::-1]

    return filtered_recipes.iloc[sorted_indices]


def substitute_ingredients(ingredients, health_conditions):
    """Provides alternative ingredients based on health conditions."""
    substitutions = {
        "sugar": "honey",
        "white sugar": "honey",
        "brown sugar": "honey",
        "butter": "olive oil",
        "margarine": "avocado oil",
        "salt": "herbs or potassium salt",
        "flour": "almond flour",
        "white flour": "whole wheat flour",
        "pasta": "zucchini noodles",
        "rice": "quinoa",
        "bread": "whole grain bread",
        "cheese": "low-fat cheese",
        "milk": "almond milk",
        "cream": "coconut cream",
        "mayonnaise": "greek yogurt",
        "sour cream": "cottage cheese",
        "red meat": "lean chicken or fish",
        "fried food": "grilled alternative",
    }

    if "Diabetes" in health_conditions:
        diabetes_subs = {
            "sugar": "stevia",
            "white sugar": "stevia",
            "brown sugar": "stevia",
            "honey": "monk fruit sweetener",
            "white bread": "whole grain bread",
            "pasta": "chickpea pasta",
            "rice": "cauliflower rice"
        }
        substitutions.update(diabetes_subs)

    if "Heart Condition" in health_conditions:
        heart_subs = {
            "butter": "olive oil",
            "margarine": "avocado oil",
            "salt": "herbs or potassium salt",
            "fried food": "baked or grilled food"
        }
        substitutions.update(heart_subs)

    return [substitutions.get(ing.lower(), ing) for ing in ingredients]


def show_recipes(recommendations, health_conditions):
    """Display recommended recipes categorized into Main Dish, Side Dish, and Desserts."""
    if recommendations.empty:
        st.warning("🚨 No recipes match your preferences. Try adjusting your filters!")
        return  # Stop execution if no recipes are found

    # Categorizing recipes based on `tags_cleaned`
    categories = {
        "Main Dish": [],
        "Side Dish": [],
        "Desserts": []
    }

    for _, row in recommendations.iterrows():
        # Convert 'tags_cleaned' into a list safely
        tags = str(row['tags_cleaned']).lower().split()  # Convert to lowercase and split into individual tags

        if any(tag in tags for tag in ["main-dish", "main course"]):
            categories["Main Dish"].append(row)
        elif any(tag in tags for tag in ["side-dish", "side course"]):
            categories["Side Dish"].append(row)
        elif any(tag in tags for tag in ["desserts", "cake", "pastry"]):
            categories["Desserts"].append(row)
        else:
            categories["Main Dish"].append(row)  # Default to Main Dish

    # Function to display recipes in each category
    def display_recipes(category_name, recipes, icon):
        if not recipes:  # Show message if no recipes are available in the category
            st.info(f"⚠️ No {category_name.lower()} recipes found.")
            return

        # **📝 Bigger Font Size for Categories with Icons**
        st.markdown(
            f'<h1 style="font-size:30px; text-align:left;">{icon} {category_name}</h1><hr>',
            unsafe_allow_html=True
        )
        for index, row in enumerate(recipes):
            recipe_name = row['name']
            recipe_id = row['id']
            poster_url = fetch_poster(recipe_name, recipe_id)

            # Unique Checkbox Key (Avoid Duplicate Keys)
            unique_key = f"recipe_{category_name}_{index}"

            selected = st.checkbox(
                f"✅ {recipe_name}",
                key=unique_key,
                value=recipe_name in st.session_state.selected_recipes
            )

            if selected:
                st.session_state.selected_recipes[recipe_name] = row.to_dict()
            else:
                st.session_state.selected_recipes.pop(recipe_name, None)

            save_selected_recipes()  # Save selections

            # Display recipe details
            st.markdown(f"### 🍽️ **{recipe_name.upper()}**")

            if poster_url:
                st.image(poster_url, caption=recipe_name, use_container_width=True)
            else:
                st.image("https://via.placeholder.com/500", caption="Image not available", use_container_width=True)

            # Ingredients
            original_ingredients = eval(row['ingredients'])
            modified_ingredients = substitute_ingredients(original_ingredients, health_conditions)

            st.markdown("### 🥕 Ingredients:")
            for original, modified in zip(original_ingredients, modified_ingredients):
                if original.lower() != modified.lower():
                    st.markdown(
                        f"- **{original}** → <span style='color: green;'>✔️ {modified}</span>", 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"- {original}")

            # Steps (Added Section)
            steps = eval(row['steps'])
            st.markdown("### 📝 Steps:")
            for step_index, step in enumerate(steps, start=1):
                st.markdown(f"{step_index}. {step}")

            st.markdown("---")


    # Display categorized recipes with icons
    display_recipes("Main Dish", categories["Main Dish"], "🍛")
    display_recipes("Side Dish", categories["Side Dish"], "🥗")
    display_recipes("Desserts", categories["Desserts"], "🍰")


def show_selected_recipes():

    if not st.session_state.selected_recipes:
        st.warning("You haven't selected any recipes yet!")
        return

    for row in st.session_state.selected_recipes.values():
        recipe_name = row['name']
        st.markdown(f"## 🍽️ **{recipe_name.upper()}**")

        # Display recipe details
        poster_url = fetch_poster(recipe_name, row['id'])
        if poster_url:
            st.image(poster_url, caption=recipe_name, use_container_width=True)
        else:
            st.image("https://via.placeholder.com/500", caption="Image not available", use_container_width=True)

        # Ingredients
        st.markdown("### 🥕 Ingredients:")
        st.markdown("\n".join([f"- {ing}" for ing in eval(row['ingredients'])]))

        # Steps
        steps = eval(row['steps'])
        st.markdown("### 📝 Steps:")
        for i, step in enumerate(steps, 1):
            st.markdown(f"{i}. {step}")

        st.markdown("---")


def parse_nutrition(nutrition_str):
    """Parse nutrition data into exactly 7 values (handles missing/excess)."""
    try:
        values = json.loads(nutrition_str.replace("'", "\""))  # Convert single quotes to double for JSON
        if isinstance(values, list):
            return (values + [0] * 7)[:7]  # Ensure exactly 7 values
    except:
        pass
    return [0, 0, 0, 0, 0, 0, 0]  # Default if parsing fails


def knapsack_select_recipes(recipes, max_calories=500, max_fat=15, max_sodium=10, min_protein=5, max_recipes=10):
    """
    Uses a greedy Knapsack approach to select recipes that optimize nutritional balance.
    Prioritizes high protein, low fat, low sugar, and moderate calories.
    """
    selected_recipes = []
    remaining_calories = max_calories

    # Extract nutrition info safely
    recipes["nutrition_values"] = recipes["nutrition"].apply(parse_nutrition)

    # Calculate protein-to-calories ratio (avoid div by zero)
    recipes["protein_to_calories"] = recipes["nutrition_values"].apply(lambda x: x[4] / (x[0] + 1))  

    # Sort recipes based on protein-to-calories ratio (higher = better)
    sorted_recipes = recipes.sort_values(by="protein_to_calories", ascending=False)

    for _, row in sorted_recipes.iterrows():
        nutrition = row["nutrition_values"]
        calories, fat, sugar, sodium, protein, saturated_fat, fiber = nutrition

        # **Check constraints**
        if calories <= remaining_calories and fat <= max_fat and sodium <= max_sodium and protein >= min_protein:
            selected_recipes.append(row.to_dict())
            remaining_calories -= calories

        # **Break conditions**
        if remaining_calories <= 0 or len(selected_recipes) >= max_recipes:
            break  # Stop if calorie limit is reached or max recipes are selected

    return selected_recipes


def show_healthy_recipes():
    """Displays healthy recipes based on user preferences and nutritional balance using Knapsack selection."""

    st.subheader("🥗 Healthy Recipes (Personalized & Nutritious)")

    # **Sidebar for User Preferences**
    st.sidebar.header("Healthy Recipe Preferences")
    diet = st.sidebar.selectbox("Dietary Preferences", ["Any", "Vegetarian", "Vegan", "Gluten-Free", "Keto", "Paleo"])
    cuisine = st.sidebar.selectbox("Cuisine", ["Any", "Italian", "Mexican", "Indian", "Chinese", "Mediterranean"])
    taste = st.sidebar.multiselect("Taste Preferences", ["Spicy", "Sweet", "Savory", "Sour"])
    included = [ing.strip() for ing in st.sidebar.text_input("Ingredients to Include (comma-separated)").split(',') if ing.strip()]
    excluded = [ing.strip() for ing in st.sidebar.text_input("Ingredients to Exclude (comma-separated)").split(',') if ing.strip()]
    additional_prefs = st.sidebar.text_area("Additional Preferences (optional)").strip()
    health_conditions = st.sidebar.multiselect("Health Conditions", ["Diabetes", "Heart Condition"])

    # **Collect User Preferences**
    preferences = [diet, cuisine] + taste + included
    preferences = [p for p in preferences if p != "Any"]

    # **Button to Search Healthy Recipes**
    if st.sidebar.button("🔍 Find Healthy Recipes"):

        # **Load Recipes**
        recipes, vectorizer, tfidf_matrix = load_models_and_data()

        # **Filter Recipes Based on User Preferences**
        filtered_recipes = recommend_recipes(preferences, included, excluded, additional_prefs, recipes, vectorizer, tfidf_matrix)

        # **Run Knapsack Selection for Balanced Nutrition**
        healthy_recipes = knapsack_select_recipes(filtered_recipes)

        if not healthy_recipes:
            st.warning("⚠️ No healthy recipes found based on your preferences. Try adjusting filters!")
            return

        # **Display Healthy Recipes**
        for row in healthy_recipes:
            recipe_name = row['name']
            recipe_id = row['id']
            poster_url = fetch_poster(recipe_name, recipe_id)

            st.markdown(f"## 🥗 {recipe_name}")

            if poster_url:
                st.image(poster_url, caption=recipe_name, use_container_width=True)
            else:
                st.image("https://via.placeholder.com/500", caption="Image not available", use_container_width=True)

            # **Show Nutrition Breakdown**
            nutrition = row["nutrition_values"]
            calories, fat, sugar, sodium, protein, saturated_fat, fiber = nutrition  

            st.markdown("### 🍎 Nutritional Information:")
            st.markdown(f"- **Calories:** {calories} kcal")
            st.markdown(f"- **Protein:** {protein} g  ✅ *Boosts muscle & metabolism*")
            st.markdown(f"- **Fiber:** {fiber} g  ✅ *Good for digestion*")
            st.markdown(f"- **Total Fat:** {fat} g")
            st.markdown(f"- **Saturated Fat:** {saturated_fat} g")
            st.markdown(f"- **Sugar:** {sugar} g ⚠️ *Lower is better*")
            st.markdown(f"- **Sodium:** {sodium} mg ⚠️ *Avoid excess sodium*")

            # **Show Ingredients with Healthier Substitutes**
            original_ingredients = eval(row['ingredients'])
            modified_ingredients = substitute_ingredients(original_ingredients, health_conditions)

            st.markdown("### 🥕 Ingredients:")
            for original, modified in zip(original_ingredients, modified_ingredients):
                if original.lower() != modified.lower():
                    st.markdown(
                        f"- **{original}** → <span style='color: green;'>✔️ {modified}</span>", 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"- {original}")

            # **Show Steps**
            steps = eval(row['steps'])
            st.markdown("### 📝 Steps:")
            for i, step in enumerate(steps, 1):
                st.markdown(f"{i}. {step}")

            st.markdown("---")

def show_ingredient_based_recipes(recipes, vectorizer, tfidf_matrix):

    """Displays recipes based on user-inputted ingredients."""
    st.sidebar.header("Find Recipes by Ingredients")
    input_ingredients = [ing.strip() for ing in st.sidebar.text_input("Enter Ingredients (comma-separated)").split(',') if ing.strip()]

    if st.sidebar.button("🔍 Search Recipes"):
        recommendations = recommend_recipes([], input_ingredients, [], "", recipes, vectorizer, tfidf_matrix)

        st.subheader("🍲 Recipes Based on Your Ingredients")

        if not recommendations.empty:  # ✅ FIXED: Removed () from `.empty`
            for index, row in recommendations.iterrows():
                recipe_name = row['name']
                recipe_id = row['id']
                poster_url = fetch_poster(recipe_name, recipe_id)

                st.markdown(f"## 🍽️ {recipe_name}")

                if poster_url:
                    st.image(poster_url, caption=recipe_name, use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/500", caption="Image not available", use_container_width=True)

                # **Show Ingredients**
                original_ingredients = eval(row['ingredients'])
                st.markdown("### 🥕 Ingredients:")
                st.markdown("\n".join([f"- {ing}" for ing in original_ingredients]))

                # **Show Steps**
                steps = eval(row['steps'])
                st.markdown("### 📝 Steps:")
                for i, step in enumerate(steps, 1):
                    st.markdown(f"{i}. {step}")

                st.markdown("---")

        else:
            st.warning("⚠️ No recipes found! Try adding different ingredients.")



def main():
    st.markdown("<h1 style='text-align: center;'>🍽️ Recipe Recommendation System</h1><hr>", unsafe_allow_html=True)

    # Navigation tabs
    tab = st.radio("Navigation", ["Preferences-Based Recipes", "Healthy Recipes", "Ingredient-Based Recipes", "Selected Recipe List"], horizontal=True)

    # Initialize variables
    recommendations = pd.DataFrame()  # Default empty DataFrame
    recipes, vectorizer, tfidf_matrix = load_models_and_data()  # Load models

    if tab == "Preferences-Based Recipes":
        # Standard sidebar for Preferences-Based Filtering
        st.sidebar.header("Your Preferences")
        diet = st.sidebar.selectbox("Dietary Preferences", ["Any", "Vegetarian", "Vegan", "Gluten-Free", "Keto", "Paleo"])
        cuisine = st.sidebar.selectbox("Cuisine", ["Any", "Italian", "Mexican", "Indian", "Chinese", "Mediterranean"])
        taste = st.sidebar.multiselect("Taste Preferences", ["Spicy", "Sweet", "Savory", "Sour"])
        included = [ing.strip() for ing in st.sidebar.text_input("Ingredients to Include (comma-separated)").split(',') if ing.strip()]
        excluded = [ing.strip() for ing in st.sidebar.text_input("Ingredients to Exclude (comma-separated)").split(',') if ing.strip()]
        additional_prefs = st.sidebar.text_area("Additional Preferences (optional)").strip()
        health_conditions = st.sidebar.multiselect("Health Conditions", ["Diabetes", "Heart Condition"])

        preferences = [diet, cuisine] + taste + included
        preferences = [p for p in preferences if p != "Any"]

        if st.sidebar.button("Recommend Recipes"):
            recommendations = recommend_recipes(preferences, included, excluded, additional_prefs, recipes, vectorizer, tfidf_matrix)

        st.subheader("🎯 Recommended Recipes Based on Your Preferences")
        if not recommendations.empty:
            show_recipes(recommendations, health_conditions)
        else:
            st.warning("No recommendations yet! Click **Recommend Recipes** in the sidebar.")

    elif tab == "Healthy Recipes":
        st.sidebar.header("Healthy Recipe Filters")
        show_healthy_recipes()  # Function to display healthy recipes

    elif tab == "Ingredient-Based Recipes":
        recipes, vectorizer, tfidf_matrix = load_models_and_data()
        show_ingredient_based_recipes(recipes, vectorizer, tfidf_matrix)

        


if __name__ == "__main__":
    main()
