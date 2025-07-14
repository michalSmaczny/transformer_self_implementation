# transformer_self_implementation
My own implementation of the Transformer architecture
https://arxiv.org/pdf/1706.03762

## Virtual Environment preparation
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
Dependency management is done with pip-tools.
```bash
pip install pip-tools
```
Dependencies used in the project should be added to `requirements.in` file.
After update of `requirements.in` file please run:
```bash
pip-compile requirements.in
```
Update your local virtual environment accordingly to current `requirements.txt` file:
```bash
pip install -r requirements.txt
```