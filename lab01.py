import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline

def read_file():
    df = pd.read_csv('data/wdbc.csv', header=None)
    rename_dict = dict([(i, f'X{i - 1}') for i in range(2, 32)])
    rename_dict[1] = 'y'
    df = df.rename(columns=rename_dict).iloc[:, 1:]
    df.iloc[:, 0] = (df.iloc[:, 0] == 'M') * 1
    return df

if __name__ == '__main__':
    df = read_file()
    train, test = train_test_split(df, test_size=0.3, random_state=42)

    # Model
    m = DecisionTreeClassifier(max_depth=3)
    m.fit(train.iloc[:, 1:], train.iloc[:, 0])

    # Predykcja
    y_pred = m.predict_proba(test.iloc[:, 1:])[:, 1]

    # Krzywa ROC
    fpr, tpr, thresholds = roc_curve(test.iloc[:, 0], y_pred)
    plt.plot(fpr, tpr)
    plt.title('Decision tree')
    # plt.show()

    # Regresja logistyczna + pipeline
    lr = LogisticRegression(max_iter=10000)
    lr.fit(train.iloc[:, 1:], train.iloc[:, 0])

    # Predykcja
    y_pred = lr.predict_proba(test.iloc[:, 1:])[:, 1]

    # Krzywa ROC
    fpr, tpr, thresholds = roc_curve(test.iloc[:, 0], y_pred)
    plt.plot(fpr, tpr)
    plt.title('Logistic regression')
    # plt.show()

    # Kroswalidacja
    cv_res = cross_val_score(lr, df.iloc[:, 1:], df.iloc[:, 0], scoring="roc_auc", cv=10)
    print(cv_res)






