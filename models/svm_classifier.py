import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

from config import Parser


class SVMClassifier:
    def __init__(self):
        parser = Parser()
        args = parser.get_section('SVM')
        self.model_path = args['svm_model_path']
        self.pipeline = Pipeline(
            [('vect', CountVectorizer(tokenizer=self.do_nothing, preprocessor=None, lowercase=False)),
             ('tfidf', TfidfTransformer()),
             ('clf', LinearSVC(class_weight='balanced'))])

    def train(self, dataset):
        X, y = dataset.df['tokens'].values, dataset.df['tag'].values
        X, y = shuffle(X, y, random_state=42)
        train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        self.pipeline.fit(train_x, train_y)

        with open(self.model_path, 'wb') as file:
            pickle.dump(self.pipeline, file)

        y_pred = self.pipeline.predict(valid_x)
        return accuracy_score(valid_y, y_pred), f1_score(valid_y, y_pred, average='macro')

    def tagging(self, dataset):
        X = dataset.df['tokens'].values

        with open(self.model_path, 'rb') as file:
            clf = pickle.load(file)

        return [clf.predict([item])[0] if item else 0 for item in X]

    @staticmethod
    # needed to pickle the object
    def do_nothing(x):
        return x
