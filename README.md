know-your-visa
==============================

A chat interface to ask questions about immigration that runs on chatgpt

# Vision AI Projects Readme Template

A simple and well designed structure is essential for any Machine Learning project, so we have designed a project template that combines **simplicity, best practice for folder structure** and **good OOP design** which could be easily scaled into any machine learning project.
The main idea is that there's much same stuff you do every time when you start your machine learning project, so wrapping all repetitive parts of documentation and delivery will help you to change just the core idea every time you start a new pytorch project. 

**So, here’s a simple template that help you get into your main project faster and just focus on your core (Model Architecture, Training Flow, etc)**

A few tips to keep in mind while using this template:
- You can remove everything above table of contents and populate project title and description if not done already.
- We are flexible with content of readme. If a section doesn't make sense for your project, remove it. Likewise, if a section is missing from the template that you would think is necessary, add it.
- Try to retain the structure of the readme, as it is the standard structure designed by the vision AI team for all vision AI projects.
- Try to not remove the "Paper Title", "In a Nutshell", "Directory structure" "Contributing", "Acknowledgments" as these apply to all projects.


# Table Of Contents
-  [Paper Title](#paper-title)
-  [In a Nutshell](#in-a-nutshell)
-  [Directory Structure](#directory-structure)
-  [Requirements](#requirements)
-  [Training](#training)
-  [Evaluation](#evaluation)
-  [Pre-trained Models](#pre-trained-models)
-  [Results](#results)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)


## Paper Title
This repository is the official implementation of [Paper Title](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

tip: use this section to link a paper if there exists or there is going to be a paper written for this project.

## In a Nutshell   
> 📋  Use this section to give the audience a quick guide on setup instructions and running a version of your project.

Instructions could include things like:

- creating appropriate folders or filepaths

-- Example: In `modeling`  folder create a python file named whatever you like, here we named it `example_model.py` . In `modeling/__init__.py` file, you can build a function named `build_model` to call your model

- creating code blocks

Example: 
```python
from .example_model import ResNet18

def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
``` 

- instructions on how to install libraries and dependencies

## Directory structure
```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── documents               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
```

>📋  This will be the actual file structure of your project which could look a bit different from the template directory structure

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training
To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation
To evaluate my model on example_dataset, run:

```eval
python eval.py --model-file mymodel.pth --benchmark example_dataset
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models
You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on <example_dataset> using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results
Our model achieves the following performance on :

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

## Future work
Discuss the scope of future work and share any literature survey already done on the topic that lead to ideas for the scope.

## Contributing
Any kind of enhancement or contribution is welcomed.

>📋  Pick a licence and describe how to contribute to your code repository.

## Acknowledgments
Acknowledge people's contribution in the project.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
