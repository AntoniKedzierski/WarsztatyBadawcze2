from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ct = ColumnTransformer([
        ('tf_idf', TfidfVectorizer(ngram_range=(1, 1)), 'statement'),
        ('speakers', OneHotEncoder(handle_unknown='ignore'), ['speaker', 'party'])
    ])

    p = Pipeline([
        ('statement', ColumnTransformer([
            ('tf_idf', TfidfVectorizer(ngram_range=(1, 1)), 'statement'),
            ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'), ['speaker', 'party', 'state'])
        ])),
        ('logreg', LogisticRegression())
    ])

    df = pd.read_csv('data/FactChecking/FactChecking/FactChecking/train.tsv', sep='\t')
    statements = df.iloc[:, 1:]
    labels = (df.iloc[:, 0] == 'pants-fire') * 1
    train_statements, test_statements, train_labels, test_labels = train_test_split(statements, labels, test_size=0.3, random_state=42)

    p.fit(train_statements, train_labels)
    predictions = p.predict_proba(test_statements)[:, 1]

    fpr, tpr, _ = roc_curve(test_labels, predictions)
    print(auc(fpr, tpr))
    plt.plot(fpr, tpr)
    plt.title('Logistic regression')
    plt.show()

    # Które słowa najważniejsze?
    coefs = p['logreg'].coef_.flatten().tolist()
    names = [n.split('__')[1] for n in p['statement'].get_feature_names_out().tolist()]

    importance = pd.DataFrame({'coef': names, 'value': coefs}).sort_values(by='value', ascending=False)
    print(importance)

