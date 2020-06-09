# Traffic Sign Network

A neural network trained on the german traffic signs benchmark dataset (GTSRB). The network is adapted from [this article](https://www.pyimagesearch.com/2019/11/04/traffic-sign-classification-with-keras-and-deep-learning/).

## Installation

### Prerequisites

* Python3 required

### Download the project

* `git clone https://github.com/grese/traffic_signs.git`
* `cd traffic_signs`

### Create a virtual environment (recommended)

* Create a venv: `python3 -m venv ./venv`
* Activate venv: `source ./venv/bin/activate`
* Install dependencies: `pip install -r requirements.txt`

### Download the dataset

* [Download the zip file](https://drive.google.com/uc?authuser=0&export=download&id=1WxQ-LmSLUUDbFaA5GGgvx1y_gP4lk1iP)
* Unzip file: `unzip ~/Downloads/gtsrb.zip -d ./data` *(after unzipping, data will be in `data/gtsrb/`)*
* Remove the zip file: `rm data/gtsrb.zip`

## Process Dataset

Loads & processes the testing and training data, and generates 5 reusable pickled object files:
(`data/X_train.pkl`, `data/y_train.pkl`, `data/X_test.pkl`, `data/y_test.pkl`, `data/signs.pkl`)

* Open `dataset.ipynb`, and run all cells (this may take several minutes)

## Build & Train Network

Builds and trains the network using the pickled objects created by `dataset.ipynb`

* Open `network.ipynb`, and run all cells.
