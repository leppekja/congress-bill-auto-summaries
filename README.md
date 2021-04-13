# congress-bill-auto-summaries

2021 Advanced ML Project to use abstract summarization on a set of Congressional Bills

## Set Up Environment

A requirements.txt file is available with the requisite packages necessary to collect the data and run the model.

## Data Collection

Data is collected from the Bulk Data Repository at govinfo.gov. The script to collect both the summaries and bill texts is available in data_collection.py; run with

    python data_collection.py

Output will include multiple .csv files for bill summaries, bill summaries with bill information, and bill summaries joined with bill texts. This script may take several hours to run.

## Pre-processing the Data
