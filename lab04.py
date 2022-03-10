from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import estimator_html_repr

import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 30)

def test_pipeline(p, train_x, train_y, test_x, test_y):
    p.fit(train_x, train_y)
    predictions = p.predict_proba(test_x)[:, 1]

    fpr, tpr, _ = roc_curve(test_y, predictions)
    print(auc(fpr, tpr))
    plt.plot(fpr, tpr)
    plt.title('Logistic regression')
    plt.show()

    with open('my_estimator.html', 'w', encoding='utf-8') as file:
        file.write(estimator_html_repr(p))


if __name__ == '__main__':
    p_best = Pipeline([
        ('column_transform', ColumnTransformer([
            ('tf_idf', TfidfVectorizer(ngram_range=(1, 1)), 'statement'),
            ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'), ['speaker', 'party', 'state'])
        ])),
        ('logreg', LogisticRegression())
    ])

    p_svd = Pipeline([
        ('column_transform', ColumnTransformer([
            ('statements', Pipeline([
                ('tf_idf', TfidfVectorizer(stop_words='english', ngram_range=(1, 1))),
                ('svd', TruncatedSVD(n_components=256, n_iter=40))
            ]), 'statement'),
            ('one_hot_encoding', OneHotEncoder(handle_unknown='ignore'), ['speaker', 'party', 'state'])
        ])),
        ('logreg', LogisticRegression())
    ])

    p_diri = Pipeline([
        ('column_transform', ColumnTransformer([
            ('statements', Pipeline([
                ('tf_idf', CountVectorizer(stop_words='english')),
                ('svd', LatentDirichletAllocation(n_components=10))
            ]), 'statement'),
            ('one_hot_encoding', OneHotEncoder(handle_unknown='ignore'), ['speaker', 'party', 'state'])
        ])),
        ('logreg', LogisticRegression())
    ])

    df = pd.read_csv('data/FactChecking/FactChecking/FactChecking/train.tsv', sep='\t')
    statements = df.iloc[:, 1:]
    labels = (df.iloc[:, 0] == 'pants-fire') * 1
    train_statements, test_statements, train_labels, test_labels = train_test_split(statements, labels, test_size=0.3, random_state=42)



    test_pipeline(p_best, train_statements, train_labels, test_statements, test_labels)


