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


def remove_punctuation(summary):
    '''
    Note remove_punctuation needs to be run prior to digit replacement.
    '''
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
    Split by word.
    '''
    return summary.str.split()


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


def process(text, *argv):
    '''
    Processes text with the given functions as arguments.
    '''
    cleaned_text = text
    for arg in argv:
        cleaned_text = arg(cleaned_text)
    return cleaned_text


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
    args = parser.parse_args()
    df = pd.read_csv(args.csv)

    if args.summaries:
        # Only remove HTML tags, allows for sentence checking
        df['summary'] = df.summary.apply(remove_html_tags)

        # Create a new column with all special characters,etc, removed.
        df['summary_clean'] = df.summary.apply(remove_html_tags)
        df['summary_clean'] = df.summary_clean.apply(lowercase)
        df['summary_clean'] = df.summary_clean.apply(remove_parenthesis_text)
        df['summary_clean'] = df.summary_clean.apply(remove_whitespace_chars)
        df['summary_clean'] = df.summary_clean.apply(remove_punctuation)
        df['summary_clean'] = df.summary_clean.apply(replace_digits)
        df['summary_clean'] = df.summary_clean.apply(shorten_spaces)
        if not args.bills:
            df.to_csv('Cleaned_Summaries.csv', index=False)

    if args.bills:
        df['bill_clean'] = df.text.apply(remove_html_tags)
        df['bill_clean'] = df.bill_clean.apply(lowercase)
        df['bill_clean'] = df.bill_clean.apply(remove_parenthesis_text)
        df['bill_clean'] = df.bill_clean.apply(remove_whitespace_chars)
        df['bill_clean'] = df.bill_clean.apply(remove_punctuation)
        df['bill_clean'] = df.bill_clean.apply(replace_digits)
        df['bill_clean'] = df.bill_clean.apply(shorten_spaces)
        df['bill_clean'] = df.bill_clean.apply(remove_bill_number)
        df['bill_clean'] = df.bill_clean.apply(remove_textxml)

        if args.summaries:
            df.to_csv('Cleaned_Summaries_And_Bills.csv', index=False)
        else:
            df.to_csv('Cleaned_Bills.csv', index=False)
