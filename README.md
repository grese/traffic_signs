# Traffic Sign Network

A neural network trained on the german traffic signs benchmark dataset (GTSRB). The network is adapted from [this article](https://www.pyimagesearch.com/2019/11/04/traffic-sign-classification-with-keras-and-deep-learning/).

## Installation

Download the project

* `git clone https://github.com/grese/traffic_signs.git`
* `cd scad_tot`

Follow these steps to setup a virtual environment (recommended).

* Create a venv: `python3 -m venv ./venv`
* Activate venv: `source ./venv/bin/activate`
* Install dependencies: `pip install -r requirements.txt`

Download the dataset

* Visit `https://drive.google.com/uc?authuser=0&export=download&id=1WxQ-LmSLUUDbFaA5GGgvx1y_gP4lk1iP` to download the zip file
* Place the zip file in `./data`, and unzip.
* After it unzipping, the directory tree should look like: `data/gtsrb/`
* Remove the zip file: `rm data/gtsrb.zip`

## Process Data

* Open `data.ipynb`, and run all cells (this may take a couple minutes)

## Build Network

* Open `network.ipynb`, and run all cells.
