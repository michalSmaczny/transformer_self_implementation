import yaml
from dataset.tokenize import _get_tokenizer_save_path
from datasets import DatasetDict
from tqdm import tqdm


def get_dataset_metadata(
    dataset: DatasetDict,
    first_lang: str,
    second_lang: str,
    vocab_size: int,
    token_model: str,
):
    metadata = dict()
    for row in tqdm(dataset["train"]):
        idx = row["idx"]
        source_len = len(row["source_ids"])
        target_len = len(row["target_ids"])
        metadata[idx] = {
            "source_len": source_len,
            "target_len": target_len,
            "max_len": max(source_len, target_len),
        }

    tokenizer_save_path = _get_tokenizer_save_path(
        first_lang=first_lang,
        second_lang=second_lang,
        vocab_size=vocab_size,
        token_model=token_model,
    )
    with open(f"{tokenizer_save_path}/token_metadata.yaml", "w") as file:
        yaml.dump(metadata, file)
