import argparse
import os
import sys

from dataset.token_counter import get_dataset_metadata
from dataset.tokenize import (
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
    parser.add_argument("--train_tokenizer", action="store_true")
    parser.add_argument("--tokenize_data", action="store_true")
    args = parser.parse_args()

    return args


args = _parse_args()

if args.train_tokenizer:
    get_data_and_train_tokenizer(
        first_lang=args.first_lang,
        second_lang=args.second_lang,
        vocab_size=args.vocab_size,
        token_model=args.token_model,
        max_sentence_length=args.max_sentence_length,
        shuffle_input_sentence=args.shuffle_input_sentence,
    )

if args.tokenize_data:
    tokenizer = load_tokenizer(
        args.first_lang, args.second_lang, args.vocab_size, args.token_model
    )

    dataset = get_tokenized_dataset(
        tokenizer=tokenizer,
        first_lang=args.first_lang,
        second_lang=args.second_lang,
    )

    get_dataset_metadata(
        dataset=dataset,
        first_lang=args.first_lang,
        second_lang=args.second_lang,
        vocab_size=args.vocab_size,
        token_model=args.token_model,
    )
