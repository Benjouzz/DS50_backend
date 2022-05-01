import pandas as pd
import mysql.connector
from mysql.connector import Error
import numpy as np
from scipy.spatial import distance as sp_dist

TAGS = ['fiction', 'fantasy', 'romance', 'classic', 'mystery', 'kindle', 'sci-fi', 'literature',
        'horror', 'contemporary', 'adventure', 'historical', 'adult', 'paranormal',
        'thriller', 'history', 'dystopia', 'audio', 'children', 'school', 'philosophy', 'novel', 'young'
]

#Encodes processed_data that need to be encoded according one book 
def encodeLabels(df, book_id, columns):
    for col in columns:
        df[col] = (df[col] == df.loc[book_id, col]).astype(float)
    return df

#Get similarity rates
def getSim(array1, array2, method='cos'):
    if method == 'cos':
        return sp_dist.cosine(array1, array2)
    if method == 'euc':
        return sp_dist.euclidean(array1, array2)
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
    
    reco = {}

    refer = data.loc[book_id].tolist()

    book_ids = data.index.tolist()
    items = data.to_numpy()

    for index, item in zip(book_ids, items):
        score = getSim(refer, item, method)
        if index != book_id:
            reco[index] = score 

    reco = sorted(reco.items(), key=lambda kv: kv[1])

    return [book_id for book_id, _ in reco[:top]]


class ContentBasedFiltering:

    def __init__(self):
        self.filter_base = None
        self.data = None
        self.process = None

    def setFilterBase(self, filter_base=None):
        if filter_base:
            self.filter_base = filter_base

    def loadData(self, top=10000):
        connection = mysql.connector.connect(
            host='ds50-mysql-do-user-9644544-0.b.db.ondigitalocean.com',
            database='ds50',
            user='ds50',
            password='AVNS_4ybSd0CoPKnCL5F',
            port = '25060'
        )
        query = f"""
            SELECT
                B.*
                ,W.author_id
            FROM
                BOOK B INNER JOIN WROTE W
                    ON B.book_id = W.book_id
            WHERE
                B.book_id = {self.filter_base}
            UNION
            SELECT
                B.*
                ,W.author_id
            FROM
                BOOK B INNER JOIN WROTE W
                    ON B.book_id = W.book_id
            LIMIT {top}
        """
        if connection.is_connected():
            df = pd.read_sql(query, connection).drop_duplicates(subset=['book_id'], keep='first')
            connection.close()
        
        df.set_index('book_id', inplace=True, drop=True)
        self.data = df


    def processData(self):
        to_keep = ['author_id','publisher','publication_year', 'format']
        self.process = self.data[to_keep].copy()

        #process year column
        self.process['publication_year'].fillna(0, inplace=True)
        self.process['publication_year'][self.process['publication_year'] < 1960] = 1960
        self.process['publication_year'] = ((self.process['publication_year'] - self.process['publication_year'].min())
                                            /(self.process['publication_year'].max()-self.process['publication_year'].min()))

        #process format column
        top_format = ['Paperback', 'Hardcover', 'ebook', 'Kindle Edition', 'Mass Market Paperback', 'Audiobook']
        self.process['format'].fillna('Undefined', inplace=True)
        self.process['format'].replace('ebook','Ebook', inplace=True)
        self.process['format'] = self.process['format'].apply(lambda x : 'Audiobook' if 'Aud' in x else x)
        self.process['format'] = self.process['format'].apply(lambda x : 'Other' if x not in top_format and x != 'Undefined' else x)
        
        #process label columns
        to_encode = ['author_id', 'publisher', 'format']
        self.process = encodeLabels(self.process, self.filter_base, to_encode)


    def filter(self, top=10, discover=False):

        connection = mysql.connector.connect(
            host='ds50-mysql-do-user-9644544-0.b.db.ondigitalocean.com',
            database='ds50',
            user='ds50',
            password='AVNS_4ybSd0CoPKnCL5F',
            port = '25060'
        )

        query = f"""
            SELECT
                T.book_id
                ,T.tag_name
                ,T.tag_count / SUM(T.tag_count) OVER(PARTITION BY T.book_id) AS "perc"
            FROM
                TAG T
            WHERE
                T.book_id IN {str(self.process.index.tolist()).replace('[','(').replace(']',')')}
        """

        if connection.is_connected(): 
            tag = pd.read_sql(query, connection).pivot_table(index='book_id', columns='tag_name', values='perc').fillna(0)
            connection.close()

        final_data = self.process.merge(tag, how='inner', left_index=True, right_index=True)

        return getBestRecommendations(self.filter_base, final_data, top=top ,method='cos')

"""
if __name__ == '__main__':
    #TEST
    filtering = ContentBasedFiltering()
    filtering.setFilterBase(filter_base=27421523)
    filtering.loadData()
    filtering.processData()
    filtering.filter()
"""