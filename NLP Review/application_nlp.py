import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import mysql.connector
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Create our own metrics to evaluate our models
def getAccuracyM1(y_test, y_pred):
    diff = abs(y_test-y_pred)
    return (diff < 2).sum()/len(diff)

precisionMinimum = 0.8
NombreDeLigne = 20_000
precision_SVM = 0.0
conteurDeCreationModele = 0 #Si ce conteur dépasse 5, ca veut dire que l'on n'a pas réussit à générer un modèle assez performant en 5 tour, arrêt du script.
#On maintient l'aléatoire
np.random.seed(500)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

#Creation de notre propre métrique d'évaluation (tolérence +/- 1)
def getAccuracyM1(y_test, y_pred):
    diff = abs(y_test-y_pred)
    return (diff < 2).sum()/len(diff)

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
                    INTERACTION
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
                    INTERACTION
            """
        df = pd.read_sql(query, connection)
    return df

# Boucle de création du modèle
while precision_SVM < precisionMinimum:
    
    if conteurDeCreationModele >= 5:
        print(f"erreur, modèle pas assez performant avec précision de {precision_SVM}")
        quit()
            
    # On obtiens les données
    data = getDataset(NombreDeLigne*(conteurDeCreationModele+1)) # ON recupère 20 000 lignes pour créer notre modèle

    #On supprime les données inutiles
    data = data[(data['review_text'].str.len() > 3) & (data['review_text'].str.len() < 3000) & (((~ data['review_text'].str.isdigit()) & (data['review_text'].str.len() != 0)) | (data['rating'] != 0))]
    #On remet les index en ordre
    data = data.reset_index()
    data = data.drop(columns=["index"])
    
    
    #On supprime les ligne de commentaire vide
    data['review_text'].dropna(inplace=True)

    #On met tout en minuscule
    data['review_text'] = [entry.lower() for entry in data['review_text']]

    #On tokenise
    data['review_text'] = [word_tokenize(entry) for entry in data['review_text']]

    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for index,entry in enumerate(data['review_text']):
        # On déclare une liste vide pour stocker les mots qui suivent les règles de cette étape
        Final_words = []
        # On initialise WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # La fonction pos_tag ci-dessous fournira le 'tag', c'est-à-dire si le mot est Nom (N) ou Verbe (V) ou autre chose.
        for word, tag in pos_tag(entry):
            # La condition ci-dessous est de vérifier les mots vides et de ne considérer que les alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # L'ensemble de mots traité final pour chaque itération sera stocké dans 'text_final'
        data.loc[index,'review_text'] = str(Final_words)
        
    #création du jeu de donnée sans les zéros
    datasans0 = data[data['rating'] != 0]

    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(datasans0['review_text'],datasans0['rating'],test_size=0.3)

    #On encode les textes
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    #On vectorise
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(data['review_text'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    
    #création, entrainement du modèle et prediction sur le jeu test
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    
    precision_SVM = getAccuracyM1(Test_Y, predictions_SVM)
    print(f"la precision est de {precision_SVM} au tour {conteurDeCreationModele}")
    conteurDeCreationModele += 1 # On incrémente le conteur

print('Score: '+str(getAccuracyM1(Test_Y, predictions_SVM)))
cm = confusion_matrix(Test_Y, predictions_SVM)
CM = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,2,3,4,5])
CM.plot()
plt.grid(False)
plt.show()

#Le modèle est alors créer et il faut l'appliquer à l'ensemble du jeu de donnée
data = getDataset(allDatset=True)
X_Tfidf = Tfidf_vect.transform(data[data['rating'] == 0]['review_text'])
predictions_sentiment_SVM = SVM.predict(X_Tfidf) # On obtient un tableaux des prédictions pour chaque élément dont le rating vaut 0

k = 0
for i in range(len(data)):
    if data.loc[i,'rating'] == 0:
        if(predictions_sentiment_SVM[k] == 0):
            data.loc[i,'rating'] = 1
        else:
            data.loc[i,'rating'] = predictions_sentiment_SVM[k]
        k += 1

#On enregistre la dataframe dans un .zip pour que gregori puisse ensuite l'importer
compression_opts = dict(method='zip',
                        archive_name='out.csv')  
data.to_csv('out.zip',compression=compression_opts) 


    