# encoder-distill (WIP)
[![pypi](https://img.shields.io/pypi/v/encoder-distill.svg)](https://pypi.python.org/pypi/encoder-distill)

Align embedding spaces of PyTorch encoders with common input types. For results see https://wandb.ai/iejmac/h14_distillation.

## Install
(Not working yet)


pip install encoder-distill

## Python examples
(No examples yet)


Checkout these examples to call this as a lib:
* [example.py](examples/example.py)

## API
TODO: make API example


Setup a virtualenv:

```
python3 -m venv .env
source .env/bin/activate
pip install -e .
```

to run tests:
```
pip install -r requirements-test.txt
```
then 
```
make lint
make test
```

You can use `make black` to reformat the code

`python -m pytest -x -s -v tests -k "dummy"` to run a specific test
