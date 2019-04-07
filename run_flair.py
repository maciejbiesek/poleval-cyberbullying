import argparse

from dataset import Dataset
from models.flair_classifier import FlairClassifier


def parse_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--text_corpus", type=str, required=False, help="Path to twitter messages in txt file")
    parent_parser.add_argument("--tag_corpus", type=str, required=False, help="Path to labels in txt file")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    # subparser for training
    subparsers.add_parser("train", parents=[parent_parser])

    # subparser for tagging
    subparsers.add_parser("tagging", parents=[parent_parser])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = FlairClassifier()
    if args.mode == "train":
        model.train()
    else:
        dataset = Dataset(args.text_corpus, args.tag_corpus, False, False, False)
        tagged = model.tagging(dataset)
        tagged = [str(tag) for tag in tagged]
        with open("results.txt", 'w') as file:
            file.write("\n".join(tagged))
