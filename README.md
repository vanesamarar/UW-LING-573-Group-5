# UW-LING-573-Group-5

## Overview
This repository contains a text summarization system for user-generated product reviews. It includes both a baseline extractive model using Term Frequency-Inverse Document Frequency (TF-IDF), and an abstractive model fine-tuned from the T5-small transformer.

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
## Dataset Usage

This project uses the [Opinosis Dataset](https://github.com/kavgan/opinosis-summarization), which is provided for **research purposes only**. Due to licensing restrictions, we **do not include the raw data or gold summaries in this repository**.

### How to Obtain the Dataset
1. Clone the dataset repository:
```bash
$ git clone https://github.com/kavgan/opinosis-summarization.git
```

2. Unzip `OpinosisDataset1.0_0.zip`

3. After unzipping, copy the `topics` and `summaries-gold` directories to your `data` and `summaries-gold`  directories:
```bash
$ cp -r opinosis-summarization/OpinosisDataset1.0_0/topics UW-LING-573-Group-5/data
$ cp -r opinosis-summarization/OpinosisDataset1.0_0/summaries-gold UW-LING-573-Group-5/summaries-gold
```

## TF-IDF Usage
### To generate summaries using the TF-IDF summarizer:
```bash
$ python tf-idf.py
```
Expected output: Saves output to a file named `summaries.json` with the structure:
```
{
  "topic_name": [
    "Top sentence 1",
    "Top sentence 2"
  ]
}
```

### To evaluate the generated summaries with ROUGE:
```bash
$ python rouge_eval.py
```
Expected output: ROUGE Recall, Precision, and F1 scores are printed to the console for each metric:
```
ROUGE-1 Recall:     #score
ROUGE-1 Precision:  #score
ROUGE-1 F1 Score:   #score
```

## T5-Small Usage
### To generate summaries using the T5 model:
Run all cells in the notebook `LING573_Model.ipynb` located in the repo root or [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chelsk5/UW-LING-573-Group-5/blob/main/LING573_Model.ipynb). This notebook:
1. Loads and preprocesses the Opinosis dataset
2. Fine-tunes the T5 model on the training data
3. Generates summaries on the test set
4. Computes ROUGE metrics
5. Saves predictions and references

### Expected output after running:
Directory t5-small-results/ will be created containing:
  - `metrics.json` — JSON file with ROUGE scores (precision, recall, and F1 measures averaged over the test set) and other evaluation metrics
  - `system_summaries/` — generated summaries from the model, saved as text files (0.txt, 1.txt, ...)
  - `model_summaries/` — corresponding gold reference summaries, saved as text files (0.A.1.txt, 1.A.1.txt, ...)

## Hyperparameter Tuning Visualization
### To visualize how different hyperparameter configurations affect model performance:
1. Set up the virtual environment
```bash
$ pip install -r requirements.txt
```

2. Optional: Create an Excel sheet with the following columns and save it as a CSV file: 
```bash
$ [Learning Rate,Training Batch Size,Epoch,Rouge1 Precision,Rouge1 Recall,Rouge1 F1,Rouge2 Precision,Rouge2 Recall,Rouge2 F1,Rougel Precision,Rougel Recall,Rougel F1]
```
> **Note:** A sample file located at /t5-small_results/hyperparameter_tuning.csv is used by the script by default. This file contains results from our tuning experiments and was used to select the optimal hyperparameter combination.

3. Run the plot script to generate a visualization
```bash
$ python hyperparameter_plot.py
```

## Acknowledgements
[Opinosis Dataset](https://github.com/kavgan/opinosis-summarization/blob/master/README.md)

[TF-IDF Reference Code](https://github.com/wangyuhsin/tfidf-text-summarization/blob/main/README.md)

[T5-small Model Card](https://huggingface.co/google-t5/t5-small)

## Contributers
- **Chelsea Kendrick**  
  University of Washington  
  chelsk5@uw.edu

- **Nina Koh**  
  University of Washington  
  nbkoh@uw.edu

- **Danielle Langford**  
  University of Washington  
  danilang@uw.edu

- **Vanesa Marar**  
  University of Washington  
  vanmarar@uw.edu

- **Haoran Zhao**  
  University of Washington  
  hjzhao@uw.edu

## License
This project is for educational purposes only and is part of coursework for UW LING 573. Please refer to the dataset sources for their specific licenses.
