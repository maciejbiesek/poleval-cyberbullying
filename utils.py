from random import randint

from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

from config import Parser


def evaluate_baselines(true_y):
    args = Parser().get_section('GENERAL')
    most_frequent_label = stats.mode(true_y).mode[0]
    most_frequent_label_baseline = [most_frequent_label for _ in range(len(true_y))]
    random_baseline = [randint(0, int(args['num_classes']) - 1) for _ in range(len(true_y))]

    return ("Baselines: \n"
            f"random acc: {accuracy_score(true_y, random_baseline):.01%}, "
            f"random F1: {f1_score(true_y, random_baseline, average='macro'):.01%}, \n"
            f"most frequent label acc: {accuracy_score(true_y, most_frequent_label_baseline):.01%}, "
            f"most frequent label F1: {f1_score(true_y, most_frequent_label_baseline, average='macro'):.01%}")
