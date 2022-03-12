# CVXOPT do optymalizacji z ograniczeniami

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

from sklearn.feature_extraction.text import CountVectorizer



class CompoundTfidf:
    def __init__(self, stop_words=None):
        self.stop_words = stop_words
        pass

    def fit(self, X, y):
        '''
        :param X: Ramka danych zwierająca słowa.
        :param y: Klasy.
        :return:
        '''
        self.count_vectorizer = CountVectorizer(stop_words=self.stop_words)
        X_count = coo_matrix(self.count_vectorizer.fit_transform(X))
        unique_labels = list(pd.DataFrame(y)['label'].unique())
        print(unique_labels)
        n_labels = pd.DataFrame(y).nunique()[0]
        label_dict = dict([(label, i) for i, label in enumerate(unique_labels)])

        '''
        Ramka zliczająca:
        słowo | dokument | klasa | liczba_wystąpień
        ------+----------+-------+-----------------
        when  | 1        | 3     | 12
        
        tf(słowo, dokument) = wystąpienia słowa w dokumencie / max. liczba wystąpień słowa
        idf(słowo, dokument, klasa) = LOG(liczba dokumentów w klasie / liczba dokumentów w klasie zawierająca dane słowo)
         
        '''

        X_count_df = pd.DataFrame(zip(X_count.row, y[X_count.row], X_count.col, X_count.data)).rename(columns={0: 'document', 1: 'label', 2: 'word', 3: 'freq'})
        tf = X_count_df.groupby(by=['label', 'word']).sum('num').drop('document', axis=1).reset_index()
        max_freq = tf.groupby(by='label').max('freq').drop('word', axis=1).reset_index().rename(columns={'freq': 'max_freq'})
        max_freq = tf.merge(max_freq, left_on='label', right_on='label')['max_freq']
        tf['freq'] /= max_freq

        idf = X_count_df.groupby(by=['label', 'word'])['document'].nunique().reset_index()
        doc_in_class = X_count_df.groupby('label')['document'].nunique().reset_index().rename(columns={'document': 'max_docs'})
        idf = idf.merge(doc_in_class, left_on='label', right_on='label')
        idf['document'] = np.log(idf['max_docs'] / (1 + idf['document']))
        idf = idf.drop('max_docs', axis=1).rename(columns={'document': 'idf'})

        tf_idf = tf.merge(idf, left_on=['label', 'word'], right_on=['label', 'word'])
        tf_idf['value'] = tf_idf['freq'] * tf_idf['idf']
        tf_idf = tf_idf.drop(['freq', 'idf'], axis=1)
        tf_idf['label'] = tf_idf['label'].apply(lambda x: label_dict[x]).astype(int)

        tf_idf_sparse = coo_matrix([tf_idf['word'], tf_idf['label'], tf_idf['value']])
        print(tf_idf_sparse)



if __name__ == '__main__':
    df = pd.read_csv('data/FactChecking/FactChecking/FactChecking/train.tsv', sep='\t')
    statements = df['statement']
    labels = df['label']

    ctfidf = CompoundTfidf(stop_words='english')
    ctfidf.fit(statements, labels)



