import os
from pathlib import Path

from flair.data import Sentence
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import EvaluationMetric

from config import Parser


class FlairClassifier:
    def __init__(self):
        parser = Parser()
        args = parser.get_section('FLAIR')
        self.corpus_path = args['flair_data_path']
        self.word_emb_path = args['flair_emb_path']
        self.model_path = args['flair_model_path']
        self.hidden_size = int(args['flair_num_hidden'])
        self.epochs = int(args['flair_epochs'])

    def train(self):
        corpus = NLPTaskDataFetcher.load_classification_corpus(Path(self.corpus_path),
                                                               test_file="test_clean_text.txt",
                                                               dev_file="dev_clean_text.txt",
                                                               train_file="train_clean_text.txt")
        embeddings = [WordEmbeddings(self.word_emb_path), FlairEmbeddings('polish-forward'),
                      FlairEmbeddings('polish-backward')]
        document_embeddings = DocumentRNNEmbeddings(embeddings, hidden_size=self.hidden_size, bidirectional=True)
        classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(),
                                    multi_label=False)
        trainer = ModelTrainer(classifier, corpus)
        trainer.train(self.model_path, evaluation_metric=EvaluationMetric.MACRO_F1_SCORE, max_epochs=self.epochs)

    def tagging(self, dataset):
        classifier = TextClassifier.load_from_file(os.path.join(self.model_path, "best-model.pt"))
        sentences = dataset.df['clean_text'].values
        results = []
        for sent in sentences:
            if sent:
                sentence = Sentence(sent)
                classifier.predict(sentence)
                result = sentence.labels[0].value
            else:
                result = 0
            results.append(result)
        return results
