### Train and predict commands

Example commands (subject to change, just for inspiration):
```
python perceptron.py -d newsgroups -f feature_name
python perceptron.py -d sst2 -f feature_name
python multilayer_perceptron.py -d newsgroups -f feature_name
```

### Commands to run unittests

Ensure that your code passes the unittests before submitting it.
The commands can be run from the root directory of the project.
```
pytest tests/test_perceptron.py
pytest tests/test_multilayer_perceptron.py
```
