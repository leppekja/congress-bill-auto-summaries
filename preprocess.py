import re
import sys
import argparse
import pandas as pd


def remove_html_tags(summary):
    '''
    Remove any HTML tags in the summaries.
    Note if used on the actual bill text, this deletes the XML structure.
    '''
    return re.sub('<.+?>', '', summary)


def lowercase(summary):
    '''
    Lowercase all words given a string.
    '''
    return summary.lower()


def remove_punctuation(summary, keep_periods=False):
    '''
    Note remove_punctuation needs to be run prior to digit replacement.
    '''
    if keep_periods:
        return re.sub('[^0-9a-zA-Z\s\.]', '', summary)
    else:
        return re.sub('[^0-9a-zA-Z\s]', '', summary)


def replace_digits(summary):
    '''
    Should be run post-remove_punctuation.
    '''
    return re.sub('\d', '#', summary)


def remove_whitespace_chars(summary):
    '''
    Remove slash n, t, r, f, and v chars; replace with space.
    Need to run shorten_spaces after this. 
    '''
    return re.sub('[\t\n\r\f\v]', ' ', summary)


def shorten_spaces(summary):
    '''
    If more than one space between characters, shortens to one.
    '''
    return re.sub('\s+', ' ', summary)


def tokenize(summary):
    '''
    Split by word. Adjusts for periods to exist in separate tokens.
    '''
    adjust_periods = re.sub('\.', ' .', summary)
    return adjust_periods.split()


def remove_parenthesis_text(summary):
    '''
    Removes parenthesis and text inside the parenthesis.
    '''
    return re.sub('\(.+\)', '', summary)


def remove_unicode_chars(summary):
    '''
    Not working!
    '''
    return re.sub(r'\\x\d+', '', summary)


def remove_bill_number(text):
    '''
    Removes the bill number indicator.
    '''
    return text[15:]


def remove_textxml(text):
    '''
    Removes the textxml flag from the bills.
    '''
    return re.sub('textxml', '', text)


def process(text, *argv, **kwargs):
    '''
    Processes text with the given functions as arguments.
    Requires kwargs tokenize and keep_periods. If not given,
    results in non-tokenized and periods kept. 
    '''
    cleaned_text = text
    try:
        make_tokens = kwargs['tokenize']
        keep_periods = kwargs['keep_periods']
    except KeyError as e:
        print('Tokenize and/or keep_periods args not passed.')
        make_tokens = False
        keep_periods = False
    cleaned_test = remove_punctuation(cleaned_text, keep_periods)
    for arg in argv:
        cleaned_text = arg(cleaned_text)

    # Add start and end of sentence tokens
    cleaned_text = '<sos> ' + cleaned_text + ' <eos>'
    if make_tokens:
        return tokenize(cleaned_text)
    else:
        return cleaned_text


def trim_dataset(df, bottom_k_pct, top_k_pct):
    '''
    Remove the top and bottom n% records from the bills and summaries.
    Expects tokenized and cleaned dataset.
    Pass in pct as decimals.
    '''
    df['summary_length'] = df.summary_clean.apply(len)
    df['bill_length'] = df.bill_clean.apply(len)
    df['summary_rank'] = df.summary_length.rank(pct=True)
    df['bill_rank'] = df.bill_length.rank(pct=True)
    cut_df = df[(df.summary_rank >= bottom_k_pct) & (df.summary_rank <= top_k_pct) & (
        df.bill_rank >= bottom_k_pct) & (df.bill_rank <= top_k_pct)]
    print('Cut ' + str(df.shape[0] - cut_df.shape[0]) + ' records.')
    print(f'New min summary length is {cut_df.summary_length.min()}')
    print(f'New max summary length is {cut_df.summary_length.max()}')
    print(f'New min bill length is {cut_df.bill_length.min()}')
    print(f'New max bill length is {cut_df.bill_length.max()}')
    del cut_df['bill_length']
    del cut_df['summary_length']
    del cut_df['summary_rank']
    del cut_df['bill_rank']
    return cut_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean dataset. \
                            If both -s and -b passed, then returns\
                            one .csv file with new summary_clean  \
                            and bill_clean columns. Minimal processing \
                            is done on the original columns to allow for \
                            the creation of sentence and overlap scores.')
    parser.add_argument("csv", help="Give the dataset as a csv")
    parser.add_argument("-s", '--summaries',
                        action="store_true", help="Clean Summaries")
    parser.add_argument(
        "-b", '--bills', action="store_true", help="Clean Bills")
    parser.add_argument(
        "-t", '--tokenize', action="store_true", help="Tokenizes the words")
    parser.add_argument(
        "-p", '--keep_periods', action="store_true", help="Retains the periods in the text")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)

    if args.summaries:
        # Only remove HTML tags, allows for sentence checking
        df['summary'] = df.summary.apply(remove_html_tags)

        # Create a new column with all special characters,etc, removed.
        df['summary_clean'] = df.summary.apply(process,
                                               args=(remove_html_tags,
                                                     lowercase,
                                                     remove_parenthesis_text,
                                                     remove_whitespace_chars,
                                                     replace_digits,
                                                     shorten_spaces,
                                                     ),
                                               tokenize=args.tokenize,
                                               keep_periods=args.keep_periods)

        if not args.bills:
            df.to_csv('Cleaned_Summaries.csv', index=False)

    if args.bills:
        df['bill_clean'] = df.text.apply(process,
                                         args=(remove_html_tags,
                                               lowercase,
                                               remove_parenthesis_text,
                                               remove_whitespace_chars,
                                               replace_digits,
                                               shorten_spaces,
                                               remove_bill_number,
                                               remove_textxml),
                                         tokenize=args.tokenize,
                                         keep_periods=args.keep_periods)
        if args.summaries:
            df.to_csv('Cleaned_Summaries_And_Bills.csv', index=False)
        else:
            df.to_csv('Cleaned_Bills.csv', index=False)
