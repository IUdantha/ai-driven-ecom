# Use the official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary files into the container
COPY preprocessed_recipes.csv vectorizer.pkl tfidf_matrix.pkl /app/

# Copy the rest of the application files
COPY . .

# Expose a port if your application requires one (e.g., Flask app runs on 5000)
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "Recipe Recommendation System.py", "--server.port=8501", "--server.address=0.0.0.0"]
