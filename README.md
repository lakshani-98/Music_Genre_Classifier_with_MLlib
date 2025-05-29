# 🎧 Music_Genre_Classifier_with_MLlib

📖 Overview
This project extends the foundation laid by Taras Matyashovskyy's binary music classifier using Apache Spark's MLlib. Instead of classifying only "Pop" and "Metal," this system scales up to classify lyrics into eight music genres:

'pop', 'country', 'blues', 'rock', 'jazz', 'reggae', 'hip hop','shoegaze'

🧠 Goals
Improve classification accuracy for multi-class genre prediction

Add support for a previously unsupported music genre

Build a user-friendly web app to classify lyrics

Provide visual feedback using pie/bar charts

Use MLlib pipelines for scalable and efficient training

📁 Datasets
1. Mendeley Dataset
Source: Publicly available (26MB)

Fields: artist_name, track_name, release_date, genre, lyrics

Genres: Pop, Country, Blues, Jazz, Reggae, Rock, Hip-Hop

2. My Dataset
A custom dataset was created with 100 songs in a new genre not present in the original dataset.

100 samples from the new genre: shoegaze

Fields: artist_name, track_name, release_date, genre, lyrics

3. Merged Dataset
Combined the above two datasets for training/testing

Used for 8-class classification

🛠️ Tools & Technologies
Apache Spark MLlib (NLP + classification pipeline)

Python

Flask


📈 ML Pipeline
Tokenization and text preprocessing

Label encoding of genres

Training with MLlib's Logistic Regression

Evaluation using an 80/20 train-test split

Model saving and integration into the web UI

🌐 Web Interface Features
Input box for pasting lyrics

Submit button to classify

Graphical result display: bar chart

Dynamic feedback on genre confidence levels
