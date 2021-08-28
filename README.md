# KITE

## Code
* [helpers.py](vis/Helpers/helpers.py) : Contains code for training models and pre-processing
* [helpers2.py](vis/Helpers/helpers2.py) : Contains actual code to generate a visualization
* [Untitled.ipynb](vis/Helpers/helpers2.py) : Experimentation with code is done here
* [views.py](vis/views.py) : Compile results from [helpers2.py](vis/Helpers/helpers2.py)

## Data
[Data](vis/Data)

## Models
[Trained embeddings of all domains](vis/Models)

## HTML
[HTML Files](vis/templates/vis)

## Evaluation Data
[CSV files to be annotated](Evaluation%20Data) (Incomplete right now)

## Requirements
[Python requirements](requirements.txt)

## Installation Instructions
* Install Python3
* Open the Command Prompt/Terminal
* Install requirements
```bash
pip install -r requirements.txt
```
OR
```bash
pip3 install -r requirements.txt
```
* Change directory to this Django project
* Apply migrations  

For Linux/MacOS
```bash
python3 manage.py migrate
```
For Windows
```bash
py -m manage.py migrate
```
* Run  

For Linux/MacOS
```bash
python3 manage.py runserver
```
For Windows
```bash
py -m manage.py runserver
```

## Links for installation
* [Python installation](https://www.python.org/downloads/)