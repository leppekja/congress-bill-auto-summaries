import re


def remove_html_tags(summary):
    '''
    Remove any HTML tags in the summaries.
    Note if used on the actual bill text, this would 
    mess up the XML structure.
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
    return re.sub('\W', '', summary)


def replace_digits(summary):
    '''
    Should be run post-remove_punctuation.
    '''
    return re.sub('\d', '#', summary)
