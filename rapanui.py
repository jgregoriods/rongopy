import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from random import choice, shuffle
from tqdm import tqdm


trainDF = pd.read_csv('rn.csv')
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char',
                                         ngram_range=(4,6),
                                         max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(trainDF['text'])

train_x,test_x,train_y,test_y = train_test_split(xtrain_tfidf_ngram_chars,
                                                 trainDF['Label'],
                                                 test_size=0.25)

clf = SVC(kernel='linear', probability=True)
clf.fit(train_x, train_y)

