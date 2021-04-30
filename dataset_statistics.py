import pandas as pd
import re
import preprocess as pr


def sentence_summaries(data, colname, csv=True):
    '''
    Note that preprocessed data may have removed periods / lowercased.
    This function relies on searching for periods, capital letters, 
    and adjusting roughly for the appearence of acronyms (u.s. being frequent).
    '''
    if csv:
        df = pd.read_csv(data)
    else:
        df = data
    df[colname] = df[colname].apply(pr.remove_html_tags)

    sentences = df[colname].apply(get_sentence_counts)
    squantiles = sentences.quantile([.25, .5, .75, .99])

    print('--------------')
    print('Mean # of Sentences: ', sentences.mean())
    print('Standard Deviation of # of Sentences: ', sentences.std())
    print('Max # of Sentences: ', sentences.max())
    print('Min # of Sentences: ', sentences.min())
    print(f'Percentiles: 25th: {squantiles[0.25]}, 50th: {squantiles[0.50]}, \
        75th: {squantiles[0.75]}, 99th: {squantiles[0.99]}')

    return None


def get_sentence_counts(text):
    return text.count('.') - len(re.findall('[a-z]\.[a-z]', text) * 2)


def token_summaries(data, colname, csv=True):
    '''
    Assumes data has already be processed.
    Prints out a basic series of statistics.
    '''
    if csv:
        df = pd.read_csv(data)
    else:
        df = data

    lengths = df[colname].apply(len)
    lquantiles = lengths.quantile([.25, .5, .75, .99])

    print('Mean Length of ' + colname + ': ', lengths.mean())
    print('Standard Deviation of ' + colname + ': ', lengths.std())
    print('Max Length of ' + colname + ': ', lengths.max())
    print('Min Length of ' + colname + ': ', lengths.min())
    print(
        f'Percentiles: 25th: {lquantiles[0.25]}, 50th: {lquantiles[0.50]}, 75th: {lquantiles[0.75]}, 99th: {lquantiles[0.99]}')
