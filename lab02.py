import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

'''
Jak oddać projekt?
Imie i nazwisko jako tytuł.
Na wierszach score'y, im większa liczba, tym większa bzdura.
'''

if __name__ == '__main__':
    df = pd.read_csv('data/FactChecking/FactChecking/FactChecking/train.tsv', sep='\t')
    statements = df['statement']
    labels = df['label']

    # Zliczanie słów
    # ngram_range: 1 - słowa, 2 - bigramy, 3 - trigramy itd., podajemy w tupli
    c = CountVectorizer(stop_words='english')
    c.fit(statements)
    statements_counted = c.transform(statements) # Zlicza i wstawia w macierz rzadką. Pierwsza współrzędna to wiersz, druga kolumna, trzecia to liczba wystąpień.
    print(statements_counted)

    # TF_IDF <3
    tf_idf = TfidfVectorizer(stop_words='english')
    tf_idf.fit(statements)
    statements_tf_idf = tf_idf.transform(statements)
    # print(tf_idf.vocabulary_) - słownik z mapowaniem słów na kolumny




