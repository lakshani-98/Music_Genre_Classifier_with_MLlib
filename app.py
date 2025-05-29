import os
from flask import Flask, request, jsonify, render_template
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, Tokenizer, StopWordsRemover, Word2Vec, VectorAssembler
import nltk
from nltk.stem import PorterStemmer
from pyspark.sql.functions import udf, explode
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql import functions as F
import os
from pyspark.ml.classification import LogisticRegressionModel

import findspark
findspark.init()

app = Flask(__name__)

import os
print(os.path.exists("logistic_regression_model"))


spark = SparkSession.builder.config("spark.ui.port", "4050").getOrCreate()

# Load the pre-trained model 
relative_path = "logistic_regression_model" 
model_path = os.path.abspath(relative_path) 

# # Load the model
model = PipelineModel.load(model_path)


def extract_probabilities(probabilities):
    return probabilities.toArray().tolist()

def map_genres(probabilities):
        genre_classes = ['pop', 'country', 'blues', 'rock', 'jazz', 'reggae', 'hip hop','shoegaze', 'None']
        return [(genre_classes[i], probabilities[i]) for i in range(len(probabilities))]
    

def predict_genre_with_probabilities(input_lyrics):
    stemmer = PorterStemmer()
    
    def apply_stem(word):
        return stemmer.stem(word)
    
    stem_udf = udf(apply_stem, StringType())
    
    input_df = spark.createDataFrame([(input_lyrics,)], ["lyrics"])

    input_df = input_df.withColumn("lyrics", F.lower(F.col("lyrics")))  # Convert to lowercase
    input_df = input_df.withColumn("lyrics", F.regexp_replace(F.col("lyrics"), "[^a-zA-Z\\s]", ""))  # Remove punctuation
    
    tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    # Tokenize, remove stopwords, and apply stemming
    input_df = tokenizer.transform(input_df)
    input_df = remover.transform(input_df)
    input_df = input_df.withColumn("word", explode(F.col("filtered_words")))
    input_df = input_df.withColumn("stemmed_word", stem_udf(F.col("word")))
    input_df = input_df.groupBy("lyrics").agg(F.collect_list("stemmed_word").alias("processed_lyrics"))

    # Use the entire pipeline for transformations and predictions
    prediction_df = model.transform(input_df)
    
    # Extract probabilities
    prediction_df = prediction_df.select("probability")

    # Register UDF to extract probabilities
    extract_probabilities_udf = udf(extract_probabilities, ArrayType(DoubleType()))
    
    # Apply the UDF to get the list of probabilities
    prediction_df = prediction_df.withColumn("probabilities", extract_probabilities_udf(F.col("probability")))

    map_genres_udf = udf(map_genres, ArrayType(ArrayType(StringType())))

    # Apply the UDF to get the mapped genre and probability pairs
    prediction_df = prediction_df.withColumn("genre_probabilities", map_genres_udf(F.col("probabilities")))
    return prediction_df.select("genre_probabilities").collect()

@app.route("/")
def home():
    return render_template("index.html") 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    lyrics = data['lyrics']
    
    predictions = predict_genre_with_probabilities(lyrics)

    print(predictions)
    
    if predictions and len(predictions) > 0 and hasattr(predictions[0], 'genre_probabilities'):
        genre_probabilities = predictions[0].genre_probabilities

        # Convert probabilities to floats and find the maximum
        probabilities_float = [(genre, float(prob)) for genre, prob in genre_probabilities]

        if probabilities_float:
            max_genre, max_prob = max(probabilities_float, key=lambda item: item[1])

            if max_prob >= 0.30:
                return jsonify({
                    'predictions': genre_probabilities,
                    'predicted_genre': max_genre
                })
            else:
                return jsonify({
                    'predictions': genre_probabilities,
                    'predicted_genre': None
                })
        else:
            return jsonify({
                'predictions': [],
                'predicted_genre': None
            })
    else:
        return jsonify({
            'predictions': [],
            'predicted_genre': None
        })

if __name__ == '__main__':
    app.run(debug=True)
