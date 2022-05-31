import os
import sys
import json
import time
from collections import defaultdict

import pandas as pd
import numpy as np

from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

STOPWORDS = set(stopwords.words("english"))

precisionMinimum = 0.8
compteurDeCreationModele = 0  # Si ce compteur dépasse max_rounds, ça veut dire que l'on n'a pas réussi à générer un modèle assez performant, arrêt du script.
max_rounds = 5
# On maintient l'aléatoire
np.random.seed(500)

# Creation de notre propre métrique d'évaluation (tolérance ± 1)
def getAccuracyM1(y_test, y_pred):
    diff = abs(y_test - y_pred)
    return (diff < 2).sum() / len(diff)

def read_json_rows(filename):
    with open(filename, "r", encoding="utf-8") as f:
        for i, row in enumerate(f):
            yield (i, json.loads(row))

def getDataset(review_in, row_limit=None):
    columns = {}
    for rowindex, row in read_json_rows(review_in):
        if row_limit is not None and rowindex > row_limit:
            break

        for colname, value in row.items():
            if colname not in columns:
                columns[colname] = [value]
            else:
                columns[colname].append(value)
    return pd.DataFrame(columns)

def writeDataset(review_out, dataframe):
    with open(review_out, "w", encoding="utf-8") as outfile:
        for row in dataframe.itertuples():
            outrow = row._asdict()
            del outrow["Index"]
            outfile.write(json.dumps(row._asdict()) + "\n")

'''
#Import the review dataset
def getDataset(NombreDeLigne = 20_000, allDatset = False):
    connection = mysql.connector.connect(
                host='ds50-mysql-do-user-9644544-0.b.db.ondigitalocean.com',
                database='ds50',
                user='ds50',
                password='AVNS_4ybSd0CoPKnCL5F',
                port = '25060'
    )
    if allDatset == False:
        for i in range(6):
            query = f"""
                SELECT
                    *
                FROM
                    REVIEW
                WHERE
                    rating = {i}
                AND
                    review_text != ''
                LIMIT
                    {NombreDeLigne//5}
            """

            if connection.is_connected():
                if not i:
                    df = pd.read_sql(query, connection)
                else:
                    new_df = pd.read_sql(query, connection)
                    df = pd.concat([df, new_df])
    else:
        query = f"""
                SELECT
                    *
                FROM
                    REVIEW
            """
        df = pd.read_sql(query, connection)
    return df
'''

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    if len(sys.argv) > 2:
        training_size = int(sys.argv[2])
    else:
        training_size = None

    review_in = os.path.join(dataset_path, "goodreads_reviews_dedup.json")
    review_out = os.path.join(dataset_path, "goodreads_reviews_dedup.json")

    starttime = time.time()
    # Boucle de création du modèle
    precision_SVM = 0.0
    while precision_SVM < precisionMinimum:
        if compteurDeCreationModele >= max_rounds:
            print(f"ERREUR : Modèle pas assez performant avec précision de {precision_SVM}")
            quit()
                
        # On obtiens les données
        print("== Loading the training dataset")
        data = getDataset(review_in, training_size*(compteurDeCreationModele+1) if training_size is not None else None) # ON recupère 20 000 lignes pour créer notre modèle

        #On supprime les données inutiles
        data = data[(data['review_text'].str.len() > 3) & (data['review_text'].str.len() < 3000) & (((~ data['review_text'].str.isdigit()) & (data['review_text'].str.len() != 0)) | (data['rating'] != 0))]
        #On remet les index en ordre
        data = data.reset_index()
        data = data.drop(columns=["index"])
        
        print("== Preprocessing the training dataset")
        print("Filtering and tokenizing")
        # On supprime les ligne de commentaire vide
        data['review_text'].dropna(inplace=True)

        # On met tout en minuscule
        data['review_text'] = [word_tokenize(entry.lower()) for entry in data['review_text']]

        tag_map = defaultdict(lambda : wordnet.NOUN)
        tag_map['J'] = wordnet.ADJ
        tag_map['V'] = wordnet.VERB
        tag_map['R'] = wordnet.ADV

        lemmatizer = WordNetLemmatizer()
        for index, entry in enumerate(data['review_text']):
            # Declaring Empty List to store the words that follow the rules for this step
            Final_words = []

            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            for word, tag in pos_tag(entry):
                # Below condition is to check for Stop words and consider only alphabets
                if word not in STOPWORDS and word.isalpha():
                    word_Final = lemmatizer.lemmatize(word, tag_map[tag[0]])
                    Final_words.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
            data.loc[index, 'review_text'] = str(Final_words)
            if (index + 1) % 100 == 0:
                print(f"\rLemmatized {index+1} / {data.shape[0]} review texts", end="")
        print(f"\rLemmatized {index+1} / {data.shape[0]} review texts")
            
        #création du jeu de donnée sans les zéros
        datasans0 = data[data['rating'] != 0]

        print("== Training the model")
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(datasans0['review_text'], datasans0['rating'], test_size=0.3)

        #On encode les textes
        print("Encoding texts")
        Encoder = LabelEncoder()
        Train_Y = Encoder.fit_transform(Train_Y)
        Test_Y = Encoder.fit_transform(Test_Y)

        #On vectorise
        print("Vectorizing")
        Tfidf_vect = TfidfVectorizer(max_features=5000)
        Tfidf_vect.fit(data['review_text'])
        Train_X_Tfidf = Tfidf_vect.transform(Train_X)
        Test_X_Tfidf = Tfidf_vect.transform(Test_X)
        
        #création, entrainement du modèle et prediction sur le jeu test
        print("Training the model")
        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        SVM.fit(Train_X_Tfidf,Train_Y)
        predictions_SVM = SVM.predict(Test_X_Tfidf)
        
        precision_SVM = getAccuracyM1(Test_Y, predictions_SVM)
        print(f"Precision after round {compteurDeCreationModele} : {precision_SVM}")
        compteurDeCreationModele += 1  # On incrémente le conteur


    #Le modèle est alors créer et il faut l'appliquer à l'ensemble du jeu de données
    print("== Loading the full dataset")
    data = getDataset(review_in)
    X_Tfidf = Tfidf_vect.transform(data[data['rating'] == 0]['review_text'])
    predictions_sentiment_SVM = SVM.predict(X_Tfidf) # On obtient un tableau des prédictions pour chaque élément dont le rating vaut 0

    print("== Applying the model")
    k = 0
    for i in range(len(data)):
        if data.loc[i,'rating'] == 0:
            if(predictions_sentiment_SVM[k] == 0):
                data.loc[i,'rating'] = 1
            else:
                data.loc[i,'rating'] = predictions_sentiment_SVM[k]
            k += 1
        if (i+1) % 100 == 0:
            print(f"\rReviews rated : {k} / {i+1}", end="")
    print(f"\rReviews rated : {k} / {i+1}")
    writeDataset(review_out, data)

    endtime = time.time()
    print(f"== Section accomplished in {endtime - starttime :.3f} seconds")