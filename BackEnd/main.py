from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import openai

app = FastAPI()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

openai.api_key = " API KEY "

class RecipeRequest(BaseModel):
    message: str

@app.post("/recette/")
def get_recipe(request: RecipeRequest):
    try:
        # Tokenization, remove stop words and punctuation
        tokens = word_tokenize(request.message)
        tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

        # Lemmatization
        lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

        # Additional information
        processed_message = ' '.join(lemmatized) + " should be Tunisian recipe"

        # Call OpenAI GPT-3.5-turbo
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=processed_message,
            max_tokens=200
        )

        # Extracting recipe from response
        recipe = response.choices[0].text.strip()

        return {"recette": recipe}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
