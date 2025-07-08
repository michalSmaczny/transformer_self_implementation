# transformer_self_implementation
My own implementation of the Transformer architecture
https://arxiv.org/pdf/1706.03762

## Virtual Environment preparation
```bash
poetry install
``` 
Unfortunately, torch installation with poetry is unstable. Thus, one should run torch installation via pip:
* for non-cuda compatible chip (MacBook):
```bash
source $(poetry env info --path)/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```
* for cuda compatibile chip (NVIDIA GPU):
```bash
source $(poetry env info --path)/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```