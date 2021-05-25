# Overview of Data Collection and Preprocessing

### Author Notes

The file `read_xmls.py` was written by Jacob Leppek as part of a prior personal project to analyze IRS-990 Forms.

### Lines of Code Counts (including docstrings)

data_collection.py : 205
dataset_statistics.py : 88
load_data.py : 145
preprocess.py : 195

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
- add start and end of sentence tags
- tokenize words

Run from the command line with:

    python preprocess.py 'file_path_to_csv.csv' -s -b -t -p

Cleaning summaries is indicated with the -s flag; for bills, include the -b flag. It is also optional to tokenize words (-t) and keep periods (-p). Not including the -t or -p flags results in non-tokenized text without periods.

Note that tokenized columns [are stored](https://stackoverflow.com/questions/23111990/pandas-dataframe-stored-list-as-string-how-to-convert-back-to-list) as strings when saved to CSV. To directly read in the tokenized columns as Python's list type, use:

    from ast import literal_eval
    import pandas as pd
    df = pd.read_csv(df_name, converters={'summary_clean': literal_eval, 'bill_clean': literal_eval})

To eliminate very long or short examples from the dataset, use:

    from preprocess import trim_dataset
    # Delete any example lower than the 1st percentile and higher than the 99%
    # percentile in the dataset in terms of length of bill or summary.
    # this also calculates and deletes any summary-bill pair
    # in which the summary is longer than the bill text
    trimmed_df = trim_dataset(df, .01, .99)

See [Data-Driven Summarization of Scientific Articles](https://arxiv.org/pdf/1804.08875.pdf) as an example of this step.

## Dataset Analysis

A collections of functions to review the dataset in the context of abstract summarization is included in dataset_statistics.py. This module provides the lengths of the summaries and bills, for example, as well as some indicators about how a summary relates to a bill.

The overlap measure is defined as the union between words in the input text and output summary over the number of words in the output summary. For the full dataset, the average overlap is 77% (plus-minus 14%). The max overlap score is 1, which means for some summaries, all of the words used were featured in the bills dataset. The minimum overlap score is 17%.

Some summaries are actually longer than the text of the bills themselves. To account for this, we only retain summary-bill pairs with a summary-length : bill-length ratio of 1 or lower.
In this set, we see that the average length of a summary is roughly 20% (plus-minus 18%) of its bill.

These functions reproduce statistics generated in [Data-Driven Summarization of Scientific Articles](https://arxiv.org/pdf/1804.08875.pdf).

## Loading Data

A custom Dataset class is available in load_data.py: BillsDataset. load_data.py also implements a train, test, validation split function, build_vocab function, as well as Dataloader creation. Split returns BillsDataset objects from train, test, and validation data. Indexing into a BillsDataset instance returns a dict of {'summary': summary_clean[idx], 'bill': bill_clean[idx]}.

    from load_data import split, build_vocab, get_dataloaders
    # 50% training data, 20% testing data, 30% validation
    train, test, validate = split(df, .5, .2, .3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build vocab for word indexes. Pass this into the get_dataloaders function.
    vocab = build_vocab(training_data, 'summary','bill')
    # Batch size of 64
    # To find max bill/summary lengths, use preprocess.trim_dataset(df, 0, 1)
    dataloaders_dict = ld.get_dataloaders(64, vocab, max_summary_length (int), max_bill_length (int), train_data= train, test_data=test, valid_data = valid)

Check dataloaders with:

    labels, features = next(iter(dataloader))
    labels.size()
    features.size()
    labels[0]
    features[0]
