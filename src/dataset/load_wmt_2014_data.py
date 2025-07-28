from datasets import DatasetDict, load_dataset


def load_wmt_14_data(source_lang: str = "de", target_lang: str = "en") -> DatasetDict:
    """
    Loads WMT 14 dataset of a chosen language pair.
    It uses HuggingFace datasets library.
    Data will be saved in data folder in this repo.
    """
    dataset = load_dataset(
        path="wmt14",
        name=f"{source_lang}-{target_lang}",
        cache_dir="./data/wmt14",
    )

    return dataset
