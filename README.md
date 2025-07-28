# transformer_self_implementation
My own implementation of the Transformer architecture
https://arxiv.org/pdf/1706.03762

### Virtual Environment preparation
Virtual environment should be created with python 3.11 or newer.
```bash
pyenv install 3.11.5
```
Creating local virtual environment dedicated to this project:
```bash
python3.11 -m venv .venv
pip install --upgrade pip
```
Virtual environment activation:
```bash
source .venv/bin/activate
```
Installing all dependencies specified in `pyproject.toml:`
```bash
pip install -e ".[dev]"
```
To dump dependencies to requirements.txt please run:
```bash
pip-compile pyproject.toml -o requirements.txt
pip-compile --extra dev -o requirements-dev.txt pyproject.toml

```
Installing pre commit hooks (linters):
```bash
pre-commit install
```
### Data Preparation
Download WMT-2014 dataset and tokenize:
```bash
python ./scripts/prepare_wmt_14_data.py \
	--first_lang=de \
	--second_lang=en \
	--vocab_size=16000 \
	--token_model=bpe \
	--max_sentence_length=8384 \
	--shuffle_input_sentence=True \
	--train_tokenizer \
	--tokenize_data
```
