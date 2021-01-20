import sys
import nltk
import numpy as np
import pandas as pd
import pickle
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import os
from matplotlib import pyplot as plt
sys.path.append(".")
sys.path.append("..")


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


column_to_predict = "ticket_type"


classifier = "NB" 
use_grid_search = False  
remove_stop_words = True  
stop_words_lang = 'english'  
use_stemming = False  
fit_prior = True  
min_data_per_class = 1  

if __name__ == '__main__':

    dfTickets = pd.read_csv(
        './datasets/all_tickets.csv',
        dtype=str
    )  

    text_columns = "body"  
    
    bytag = dfTickets.groupby(column_to_predict).aggregate(np.count_nonzero)
    tags = bytag[bytag.body > min_data_per_class].index
    dfTickets = dfTickets[dfTickets[column_to_predict].isin(tags)]


    labelData = dfTickets[column_to_predict]
    data = dfTickets[text_columns]


    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labelData, test_size=0.2
    )  


    if remove_stop_words:
        count_vect = CountVectorizer(stop_words=stop_words_lang)
    elif use_stemming:
        count_vect = StemmedCountVectorizer(stop_words=stop_words_lang)
    else:
        count_vect = CountVectorizer()

    if classifier == "NB":


        text_clf = Pipeline([
            ('vect', count_vect),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB(fit_prior=fit_prior))
        ])
        text_clf = text_clf.fit(train_data, train_labels)

    elif classifier == "SVM":
        print("Training SVM classifier")
        text_clf = Pipeline([(
            'vect', count_vect),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(
                loss='hinge', penalty='l2', alpha=1e-3,
                n_iter=5, random_state=42
            )
        )])
        text_clf = text_clf.fit(train_data, train_labels)

    if use_grid_search:

        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3)
        }

        gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(train_data, train_labels)

        gs_clf.best_score_
        gs_clf.best_params_


    predicted = text_clf.predict(test_data)
    prediction_acc = np.mean(predicted == test_labels)
    print("Confusion matrix without GridSearch:")
    print(metrics.confusion_matrix(test_labels, predicted))
    print("Mean without GridSearch: " + str(prediction_acc))

    if use_grid_search:
        predicted = gs_clf.predict(test_data)
        prediction_acc = np.mean(predicted == test_labels)
        print("Confusion matrix with GridSearch:")
        print(metrics.confusion_matrix(test_labels, predicted))
        print("Mean with GridSearch: " + str(prediction_acc))

    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import matplotlib
    mat = confusion_matrix(test_labels, predicted)
    plt.figure(figsize=(4, 4))
    sns.set()
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=np.unique(test_labels),
                yticklabels=np.unique(test_labels))
    plt.xlabel('true label')
    plt.ylabel('predicted label')

    plt.show()


    from sklearn.metrics import classification_report
    print(classification_report(test_labels, predicted,
                                target_names=np.unique(test_labels)))


    if use_grid_search:
        pickle.dump(
            gs_clf,
            open(os.path.join(
                '.', 'outputs', column_to_predict+".model"),
                'wb'
            )
        )
    else:
        pickle.dump(
            text_clf,
            open(os.path.join(
                '.', 'outputs', column_to_predict+".model"),
                'wb'
            )
        )
