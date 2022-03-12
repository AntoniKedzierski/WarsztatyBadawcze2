import spacy
import re

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import estimator_html_repr

from gensim.models.word2vec import Word2Vec

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 30)


def lemmatizer(column):
    result = []
    for spacy_obj in column:
        result.append(' '.join(f'{word.lemma_} {word.tag_}' for word in spacy_obj))
    return result


def name_entity_recog(column):
    result = []
    for spacy_obj in column:
        result.append(' '.join(entity.label_ for entity in spacy_obj.ents))
    return result


def avg_word_len(column):
    result = []
    for txt in column:
        words = re.findall(r"\w+", txt)
        n_words = len(words)
        result.append(sum(len(w) for w in words) / n_words)
    return result


def num_words(column):
    result = []
    for txt in column:
        result.append(len(re.findall(r'\w+', txt)))
    return result


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')

    df = pd.read_csv('data/FactChecking/FactChecking/FactChecking/train.tsv', sep='\t')
    df['spacy_obj'] = list(nlp.pipe(df['statement']))
    df['context'] = df['context'].fillna('')

    features = df.iloc[:, 1:]
    labels = (df.iloc[:, 0] == 'pants-fire') * 1

    print(FunctionTransformer(avg_word_len(features['statement'])))


    # Pipeline
    p = Pipeline([
        ('col_transformer', ColumnTransformer([
            ('tfidf_lemmas', Pipeline([
                ('lemmatizer', FunctionTransformer(lemmatizer)),
                ('tfidf_lem', TfidfVectorizer(stop_words='english'))
            ]), 'spacy_obj'),
            ('count_entities', Pipeline([
                ('name_ent_recog', FunctionTransformer(name_entity_recog)),
                ('count_ent', CountVectorizer())
            ]), 'spacy_obj'),
            ('subjects', CountVectorizer(token_pattern=r"([^,]+)"), 'subject'),
            ('context', TfidfVectorizer(stop_words='english'), 'context'),
            ('avg_word_len', FunctionTransformer(avg_word_len), 'statement'),
            ('num_words', FunctionTransformer(num_words), 'statement'),
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), ['state', 'party', 'speaker'])
        ])),
        ('log_reg', LogisticRegression(max_iter=6000))
    ])

    # p.fit_transform(features, labels)
    # print(p['col_transformer'].transformers_)

    cv = StratifiedKFold(10)
    print(np.mean(cross_val_score(p, features, labels, scoring='roc_auc', cv=cv)))







