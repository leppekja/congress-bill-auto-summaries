import read_xmls as rx
import xml.etree.ElementTree as ET
import preprocess as ps


def extract_summary(xml):
    '''
    Implements an extractive summary baseline comparison by pulling the 
    official title from each bill. Returns processed text for comparison.
    '''
    if type(xml) != ET.Element:
        xml = ET.fromstring(xml)

    summary = rx.search_tree(xml, 'official-title')['official-title']
    clean_summary = ps.process(summary, ps.remove_html_tags,
                               ps.lowercase,
                               ps.remove_punctuation,
                               ps.replace_digits,
                               ps.remove_whitespace_chars,
                               ps.shorten_spaces,
                               ps.remove_parenthesis_text
                               )

    return (clean_summary if clean_summary != '' else '')
