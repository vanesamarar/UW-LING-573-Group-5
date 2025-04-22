# UW-LING-573-Group-5

## Overview
This repository contains a baseline text-summarization system for review data using a Term Frequency-Inverse Document Frequency (TF-IDF) approach. 

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

## Acknowledgements
Opinosis dataset repo: 
https://github.com/kavgan/opinosis-summarization/blob/master/README.md

Reference code: 
https://github.com/wangyuhsin/tfidf-text-summarization/blob/main/README.md
