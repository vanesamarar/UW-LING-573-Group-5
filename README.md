# UW-LING-573-Group-5

## Overview
This repository contains a baseline text-summarization system for review data using a Term Frequency-Inverse Document Frequency (TF-IDF) approach. 

## Dataset Usage

This project uses the [Opinosis Dataset](https://github.com/kavgan/opinosis-summarization), which is provided for **research purposes only**. Due to licensing restrictions, we **do not include the raw data or gold summaries in this repository**.

### How to Obtain the Dataset
1. Clone the dataset repository:
```bash
git clone https://github.com/kavgan/opinosis-summarization.git
```

2. Unzip OpinosisDataset1.0_0.zip

3. After unzipping, move the `topics` and `summaries-gold` directories into your `data/` and `summaries-gold`  directories:
   ```bash
   cp -r OpinosisDataset1.0_0/topics ./data/
   cp -r OpinosisDataset1.0_0/summaries-gold ./data/
```

## Instructions for Setup
1. Clone the repository
```bash
$ git clone https://github.com/chelsk5/UW-LING-573-Group-5.git
$ cd UW-LING-573-Group-5
```

2. Create a virtual environment (for MacOS)
```bash
$ python3 -m venv .venv
$ source .venv/bin/activate 
```

3. Install project dependencies
```bash
$ pip install -r requirements.txt
```

4. For Group 5 members only: If you need to install new packages, run the following commands to ensure that we're all on the same page:
```bash
$ pip install <package-name>
$ pip freeze > requirements.txt
```

## Usage
To generate summaries using the TF-IDF summarizer:
```bash
$ python tf-idf.py
```

To evaluate the generated summaries with ROUGE:
```bash
python rouge_eval.py
```

## Acknowledgements
[Opinosis Dataset](https://github.com/kavgan/opinosis-summarization/blob/master/README.md)

[TF-IDF Reference Code](https://github.com/wangyuhsin/tfidf-text-summarization/blob/main/README.md)

## Contributers
Chelsea Kendrick, Nina Koh, Danielle Langford, Vanesa Marar, Haoran Zhao
University of Washington
{chelsk5, nbkoh, danilang, vanmarar, hjzhao}@uw.edu

## License
This project is for educational purposes only and is part of coursework for UW LING 573. Please refer to the dataset sources for their specific licenses.
