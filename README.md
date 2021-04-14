# congress-bill-auto-summaries

2021 Advanced ML Project to use abstract summarization on a set of Congressional Bills

## Set Up Environment

A requirements.txt file is available with the requisite packages necessary to collect the data and run the model.

## Data Collection

Data is collected from the Bulk Data Repository at govinfo.gov. The script to collect both the summaries and bill texts is available in data_collection.py; run from the command line with

    python data_collection.py

Output will include multiple .csv files for bill summaries, bill summaries with bill information, and bill summaries joined with bill texts. This script may take several hours to run.

## Pre-processing the Data

Pre-processing steps include:

- removing HTML tags
- lowercasing all words
- removing extraneous spaces and new lines
- removing references in parentheses
- replacing digits with #
- remove XML specific content
- remove bill number

Run from the command line with:

    python preprocess.py 'file_path_to_csv.csv' -s -b

TO DO:

- Tokenize the words in this step? See preprocess.tokenize.
- Delete summaries / bills outside the 25th to 75th percentile, or other word length? See [scientific abstract paper](https://arxiv.org/pdf/1804.08875.pdf).
