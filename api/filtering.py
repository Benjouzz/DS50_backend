import pandas as pd
import mysql.connector
from mysql.connector import Error
import numpy as np

TAGS = ['fiction', 'fantasy', 'romance', 'classic', 'mystery', 'kindle', 'sci-fi', 'literature',
        'horror', 'contemporary', 'adventure', 'historical', 'adult', 'paranormal',
        'thriller', 'history', 'dystopia', 'audio', 'children', 'school', 'philosophy', 'novel', 'young'
]

#Encodes processed_data that need to be encoded according one book 
def encodeLabels(book_id, df, columns=['author', 'publisher', 'format']):
    for col in columns:
        df[col] = (df[col] == df.loc[book_id, col]).astype(float)
    return df

#Get similarity rates
def getSim(array1, array2, method='cos'):
    if method == 'cos':
        return sp.spatial.distance.cosine(array1, array2)
    if method == 'euc':
        return sp.spatial.distance.euclidean(array1, array2)
    if method == 'pea':
        num = sum([x1*x2 for x1, x2 in zip(array1, array2)])
        denom1 = 0
        denom2 = 0
        for x1, x2 in zip(array1, array2):
            if x1 != 0 and x2 != 0:
                denom1 += x1*x1
                denom2 += x2*x2 
        denom = np.sqrt(denom1)*np.sqrt(denom2)
        if denom == 0:
            return 0
        else:
            return num/denom


def getBestRecommendations(book_id, data, top=10, method='cos'):
    best = {}
    data = encodeLabels(book_id, data)

    refer = data.loc[book_id].tolist()
    book_ids = data.index.tolist()
    items = data.to_numpy()

    for index, item in zip(book_ids, items):
        score = getSim(refer, item, method)
        if index != book_id:
            best[index] = score 

    best_reco = sorted(best.items(), key=lambda kv: kv[1])

    return [book_id for book_id, _ in best_reco[:top].items()]


class ContentBasedFiltering:

    def __init__(self):
        self.filter_base = None
        self.data = None

    def setFilterBase(self, filter_base=None):
        if filter_base:
            self.filter_base = filter_base

    def load_data(self, top=1000):
        connection = mysql.connector.connect(
            host='ds50-mysql-do-user-9644544-0.b.db.ondigitalocean.com',
            database='ds50',
            user='ds50',
            password='AVNS_4ybSd0CoPKnCL5F',
            port = '25060'
        )
        query = f"""
            SELECT
                *
            FROM
                BOOK
            WHERE
                book_id = {self.filter_base}
            UNION
            SELECT
                *
            FROM
                BOOK
            LIMIT {top}
        """
        if connection.is_connected():
            df = pd.read_sql(query, connection)
            self.data = df

    def filter(self, top=10, discover=False):
        
        processed_data = pd.DataFrame({}, columns=['book_id'])
        processed_data['book_id'] = self.data['book_id']

        connection = mysql.connector.connect(
            host='ds50-mysql-do-user-9644544-0.b.db.ondigitalocean.com',
            database='ds50',
            user='ds50',
            password='AVNS_4ybSd0CoPKnCL5F',
            port = '25060'
        )

        for book_id in self.data['book_id']:
            query = f"""
                SELECT
                    T.tag_name
                    ,T.tag_count / SUM(T.tag_count) OVER() AS "perc"
                FROM
                    BOOK B INNER JOIN TAG T
                        ON B.book_id = T.book_id
                WHERE
                    B.book_id = {book_id}
            """
        
        #TODO Clear tag table in database to fit with TAGS
        """
            if connection.is_connected(): 
                tag = pd.read_sql(query, connection).pivot_table(columns='tag_name', values='perc')
                tag['book_id'] = [book_id]
                processed_data = processed_data.merge(tag, on=['book_id'], how='left')
        
        processed_data = encodeLabels(self.filter_base, processed_data)

        return getBestRecommendations(self.filter_base, processed_data, top=top ,method='cos')
        """

if __name__ == '__main__':
    #TEST
    filtering = ContentBasedFiltering()
    filtering.setFilterBase(filter_base=27421523)
    filtering.load_data()
    filtering.filter()