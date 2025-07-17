import argparse
import os
import sys

from src.dataset.wmt_2014_data import (
    get_data_and_train_tokenizer,
    get_tokenized_dataset,
    load_tokenizer,
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_lang", type=str, required=True)
    parser.add_argument("--second_lang", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--token_model", type=str, default="bpe")
    parser.add_argument("--max_sentence_length", type=int, default=4192)
    parser.add_argument("--shuffle_input_sentence", type=bool, default=False)
    args = parser.parse_args()

    return args


args = _parse_args()

get_data_and_train_tokenizer(
    first_lang=args.first_lang,
    second_lang=args.second_lang,
    vocab_size=args.vocab_size,
    token_model=args.token_model,
    max_sentence_length=args.max_sentence_length,
    shuffle_input_sentence=args.shuffle_input_sentence,
)

tokenizer = load_tokenizer(
    args.first_lang, args.second_lang, args.vocab_size, args.token_model
)

_ = get_tokenized_dataset(
    tokenizer=tokenizer,
    first_lang=args.first_lang,
    second_lang=args.second_lang,
    vocab_size=args.vocab_size,
    token_model=args.token_model,
)
