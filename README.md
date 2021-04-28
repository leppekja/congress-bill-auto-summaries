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
- tokenize words

Run from the command line with:

    python preprocess.py 'file_path_to_csv.csv' -s -b -t -p

Cleaning summaries is indicated with the -s flag; for bills, include the -b flag. It is also optional to tokenize words (-t) and keep periods (-p). Not including the -t or -p flags results in non-tokenized text with periods.

Note that tokenized columns [are stored](https://stackoverflow.com/questions/23111990/pandas-dataframe-stored-list-as-string-how-to-convert-back-to-list) as strings when saved to CSV. To directly read in the tokenized columns as Python's list type, use:

    df = pd.read_csv(df_name, converters={'summary_clean': eval, 'bill_clean':eval})

TO DO:

- Delete summaries / bills outside the 25th to 75th percentile, or other word length? See [scientific abstract paper](https://arxiv.org/pdf/1804.08875.pdf).

A custom Dataset class is available in load_data.py. Using the BillsDataset class implements a train, test, validation split method as well as Dataloaders. Indexing into a BillsDataset instance returns a dict of {summary_clean, bill_clean}.

    from load_data import BillsDataset
    data = BillsDataset('Cleaned_Summaries_And_Bills.csv', '', 'summary_clean','bill_clean')
    # 50% training data, 20% testing data, 30% validation
    data.split(.5, .2, .3)
    # Batch size of 64
    train_dataloader, test_dataloader, validate_dataloader = data.get_dataloaders(64, collate_fn=None)

## Baseline

We implement an extractive summary method that pulls the official-title section from the bill as a comparable baseline for our abstractive model. This may be found in extractive_summary.py, and uses the structure of the XML file to parse the text from the bill.

## References

#### Data Preprocessing

- [Stanford NLP Toolkit](https://www.aclweb.org/anthology/P14-5010.pdf)

#### Baseline Models

#### Models

- [Get To The Point: Summarization with POinter-Generator Networks](https://www.aclweb.org/anthology/P17-1099.pdf)
- [Summary Level Training of Sentence Rewriting for Abstractive Summarization](https://www.aclweb.org/anthology/D19-5402.pdf)
- [Abstractive Summarization Methods](https://medium.com/sciforce/towards-automatic-summarization-part-2-abstractive-methods-c424386a65ea)
- [Long Short-Term Memory Based Recurrent Neural Network](https://arxiv.org/pdf/1402.1128.pdf)
- [Text Summarization Using Neural Networks](https://digital.library.txstate.edu/bitstream/handle/10877/3819/fulltext.pdf)
- [Neural Text Summarization](https://cs224d.stanford.edu/reports/urvashik.pdf)
- [Text Summarization using RNN](https://iq.opengenus.org/text-summarization-using-rnn/)
