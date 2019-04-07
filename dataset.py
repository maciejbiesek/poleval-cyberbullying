import os
import pickle
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from spacy.lang.pl import Polish

from config import Parser

RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)


class Dataset:
    def __init__(self, texts_file, tags_file, clean_data=True, remove_stopwords=False, is_train=True):
        self.args = Parser().get_sections(['GENERAL', 'RNN', 'FLAIR'])
        self.max_sent_length = int(self.args['max_sent_length'])
        self.batch_size = int(self.args['batch_size'])
        self.emb_size = int(self.args['emb_size'])
        self.clean_data = clean_data
        self.remove_stopwords = remove_stopwords
        self.is_train = is_train

        self.nlp = Polish()
        self.df = self.build_dataframe(texts_file, tags_file)
        self.unk_emb = self.get_random_emb(self.emb_size)
        self.word2idx, self.idx2word = self.build_dict()
        if self.is_train:
            self.embeddings = self.get_embeddings(self.args['emb_path'])

    def build_dataframe(self, texts_file, tags_file):
        with open(texts_file) as file:
            lines = [line.strip() for line in file.readlines()]
            texts = pd.DataFrame(lines, columns=['text'])
        tags = pd.read_fwf(tags_file, header=None, names=['tag'])
        df = pd.concat([texts, tags], axis=1)
        df['tokens'] = df['text'].map(lambda x: self.preprocess_sentence(x))
        df['length'] = df['tokens'].map(lambda x: len(x))
        df['clean_text'] = df['tokens'].map(lambda x: " ".join(x))
        if self.clean_data:
            df = self.clean(df)
        return df

    def preprocess_sentence(self, sentence):
        sentence = sentence.replace(r"\n", "").replace(r"\r", "").replace(r"\t", "").replace("„", "").replace("”", "")
        doc = [tok for tok in self.nlp(sentence)]
        if not self.clean_data and str(doc[0]) == "RT":
            doc.pop(0)
        while str(doc[0]) == "@anonymized_account":
            doc.pop(0)
        while str(doc[-1]) == "@anonymized_account":
            doc.pop()
        if self.remove_stopwords:
            doc = [tok for tok in doc if not tok.is_stop]
        doc = [tok.lower_ for tok in doc]
        doc = ["".join(c for c in tok if not c.isdigit() and c not in string.punctuation) for tok in doc]
        doc = [RE_EMOJI.sub(r'', tok) for tok in doc]
        doc = [tok.strip() for tok in doc if tok.strip()]
        return doc

    def build_dict(self):
        if self.is_train:
            sentences = self.df['tokens']
            all_tokens = [token for sentence in sentences for token in sentence]
            words_counter = Counter(all_tokens).most_common()
            word2idx = {
                self.args['pad']: 0,
                self.args['unk']: 1
            }
            for word, _ in words_counter:
                word2idx[word] = len(word2idx)

            with open(self.args['word_dict_path'], 'wb') as dict_file:
                pickle.dump(word2idx, dict_file)

        else:
            with open(self.args['word_dict_path'], 'rb') as dict_file:
                word2idx = pickle.load(dict_file)

        idx2word = {idx: word for word, idx in word2idx.items()}
        return word2idx, idx2word

    def transform_dataset(self):
        sentences = self.df['tokens'].values
        x = [sentence[:self.max_sent_length] for sentence in sentences]
        x = [sentence + [self.args['pad']] * (self.max_sent_length - len(sentence)) for sentence in x]
        x = [[self.word2idx.get(word, self.word2idx[self.args['unk']]) for word in sentence] for sentence in x]
        y = self.df['tag'].values
        return np.array(x), np.array(y)

    def parse_dataset(self):
        x, y = self.transform_dataset()
        if self.is_train:
            x, y = shuffle(x, y, random_state=42)
            train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)
            return list(self.chunks(train_x, train_y, self.batch_size)), valid_x, valid_y
        return list(self.chunks(x, y, self.batch_size))

    def get_embeddings(self, embeddings_file):
        emb_list = []
        print("Loading vectors...")
        word_vectors = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
        print("Vectors loaded...")
        for _, word in sorted(self.idx2word.items()):
            if word == self.args['pad']:
                word_vec = np.zeros(self.emb_size)
            elif word == self.args['unk']:
                word_vec = self.unk_emb
            else:
                try:
                    word_vec = word_vectors.word_vec(word)
                except KeyError:
                    word_vec = self.unk_emb
            emb_list.append(word_vec)
        return np.array(emb_list, dtype=np.float32)

    def get_class_weight(self):
        y = self.df['tag'].values
        _, counts = np.unique(y, return_counts=True)
        return np.array(1 - counts / y.size)

    def prepare_flair_format(self, column_name):
        X = self.df[column_name].values
        y = self.df['tag'].values

        train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.15, random_state=42,
                                                          stratify=train_y)
        train_data = [f"__label__{label} {text}" for text, label in zip(train_x, train_y)]
        val_data = [f"__label__{label} {text}" for text, label in zip(val_x, val_y)]
        dev_data = [f"__label__{label} {text}" for text, label in zip(dev_x, dev_y)]
        flair_dir_path = self.args['flair_data_path']
        os.makedirs(flair_dir_path, exist_ok=True)

        with open(os.path.join(flair_dir_path, f"train_{column_name}.txt"), 'w') as train_file, \
                open(os.path.join(flair_dir_path, f"dev_{column_name}.txt"), 'w') as dev_file, \
                open(os.path.join(flair_dir_path, f"test_{column_name}.txt"), 'w') as val_file:
            train_file.write("\n".join(train_data))
            dev_file.write("\n".join(dev_data))
            val_file.write("\n".join(val_data))

    def print_stats(self):
        print(self.df['length'].describe())
        print(self.df['length'].quantile(0.95, interpolation='lower'))
        print(self.df['length'].quantile(0.99, interpolation='lower'))
        print(self.df.shape)
        print(self.df['tag'].value_counts())

    @staticmethod
    def get_random_emb(length):
        return np.random.uniform(-0.25, 0.25, length)

    @staticmethod
    def clean(dataframe):
        dataframe = dataframe.drop_duplicates('clean_text')
        return dataframe[(dataframe['tokens'].apply(lambda x: "rt" not in x[:1])) & (dataframe['length'] > 1)]

    @staticmethod
    def chunks(inputs, outputs, batch_size):
        for i in range(0, len(inputs), batch_size):
            yield inputs[i:i + batch_size], outputs[i:i + batch_size]


if __name__ == "__main__":
    dt = Dataset("../data/task_6-2/training_set_clean_only_text.txt",
                 "../data/task_6-2/training_set_clean_only_tags.txt",
                 True, False, False)
    # dt.prepare_flair_format("clean_text")
