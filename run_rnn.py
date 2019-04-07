import argparse

from config import Parser
from dataset import Dataset
from models.gru_classifier import GRU
from utils import evaluate_baselines


def parse_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--text_corpus", type=str, required=True, help="Path to twitter messages in txt file")
    parent_parser.add_argument("--tag_corpus", type=str, required=True, help="Path to labels in txt file")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    # subparser for training
    subparsers.add_parser("train", parents=[parent_parser])

    # subparser for tagging
    subparsers.add_parser("tagging", parents=[parent_parser])

    return parser.parse_args()


def train(dataset):
    model = GRU(dataset)
    model.train()


if __name__ == "__main__":
    args = parse_args()
    config_args = Parser().get_section('RNN')
    if args.mode == "train":
        dataset = Dataset(args.text_corpus, args.tag_corpus, True, False, True)
        train(dataset)
    else:
        dataset = Dataset(args.text_corpus, args.tag_corpus, False, False, False)
        model = GRU(None, False)
        tagged = model.tagging(dataset)
        tagged = [str(tag) for tag in tagged]
        with open("results.txt", 'w') as file:
            file.write("\n".join(tagged))
