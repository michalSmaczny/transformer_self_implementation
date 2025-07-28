import os
from typing import Dict, Iterator

import sentencepiece as spm
from dataset.load_wmt_2014_data import load_wmt_14_data
from datasets import DatasetDict


def _generator_wmt_14_data(first_lang: str, second_lang: str) -> Iterator[str]:
    dataset = load_wmt_14_data(source_lang=first_lang, target_lang=second_lang)
    train_data = dataset["train"]
    for row in train_data:
        yield row["translation"][first_lang]
        yield row["translation"][second_lang]


def get_tokenizer_name(
    first_lang: str, second_lang: str, vocab_size: int, token_model: str
) -> str:
    return f"{first_lang}_{second_lang}_{vocab_size}_{token_model}"


def get_data_and_train_tokenizer(
    first_lang: str,
    second_lang: str,
    vocab_size: int,
    token_model: str,
    max_sentence_length: int,
    shuffle_input_sentence: bool,
) -> None:
    """
    Downloads or loads WMT 14 dataset (via generator).
    Then runs SentencePiece tokenizer on this dataset.
    Vocab is saved to tokenizers folder in this repo.
    """
    wmt_2014_gen = _generator_wmt_14_data(first_lang, second_lang)
    tokenizer_name = get_tokenizer_name(
        first_lang, second_lang, vocab_size, token_model
    )
    folder_path = f"./tokenizers/{tokenizer_name}"
    os.makedirs(folder_path, exist_ok=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=wmt_2014_gen,
        model_prefix=f"{folder_path}/{tokenizer_name}",
        vocab_size=vocab_size,
        model_type=token_model,
        character_coverage=1.0,
        max_sentence_length=max_sentence_length,
        shuffle_input_sentence=shuffle_input_sentence,
    )


def _get_tokenizer_save_path(
    first_lang: str,
    second_lang: str,
    vocab_size: int,
    token_model: str,
    path_to_file: bool = False,
) -> str:
    tokenizer_name = get_tokenizer_name(
        first_lang, second_lang, vocab_size, token_model
    )
    save_path = f"./tokenizers/{tokenizer_name}"
    if path_to_file:
        save_path = f"{save_path}/{tokenizer_name}.model"

    return save_path


def load_tokenizer(
    first_lang: str, second_lang: str, vocab_size: int, token_model: str
) -> spm.SentencePieceProcessor:
    tokenizer_save_path = _get_tokenizer_save_path(
        first_lang, second_lang, vocab_size, token_model, path_to_file=True
    )
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_save_path)

    return tokenizer


def _tokenize_function(example, tokenizer) -> Dict[str, str]:
    source_ids = tokenizer.encode(example["translation"]["en"])
    target_ids = tokenizer.encode(example["translation"]["de"])

    return {"source_ids": source_ids, "target_ids": target_ids}


def _add_index_column(dataset: DatasetDict) -> DatasetDict:
    for name, dataset_tbl in dataset.items():
        idxs_list = list(range(len(dataset_tbl)))
        dataset_tbl = dataset_tbl.add_column("idx", idxs_list)
        dataset[name] = dataset_tbl

    return dataset


def get_tokenized_dataset(
    tokenizer: spm.SentencePieceProcessor,
    first_lang: str,
    second_lang: str,
) -> DatasetDict:
    """
    Retrieves the tokenized dataset.
    If a cached version isn't found, it tokenizes the data using a pre-trained
    tokenizer.
    """
    dataset = load_wmt_14_data(source_lang=first_lang, target_lang=second_lang)
    tokenized_dataset = dataset.map(
        _tokenize_function,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["translation"],
    )
    tokenized_dataset = _add_index_column(tokenized_dataset)

    return tokenized_dataset
