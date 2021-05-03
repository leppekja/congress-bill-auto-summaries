import re
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
import preprocess as ps
import read_xmls as rx


def extract_summary(xml):
    '''
    Implements an extractive summary baseline comparison by pulling the 
    official title from each bill. Returns processed text for comparison.
    Note that a number of XML documents have formatting issues. In these
    cases, no summary is returned (if the document cannot be read) (1 found),
    or an abriged summary is returned (in cases where references within the 
    official-title field break the text off.) (~300 found)
    The latter issue occurs when an external-xref tag and parsable citation
    is used, and can be located by filtering extracted summaries where 
    str.len() < 15. A workaround to obtain these summaries
    is checked for in this function in solve_citation_issue.
    '''
    if type(xml) != ET.Element:
        try:
            xml = ET.fromstring(xml)
        except ParseError:
            return ''

    summary = rx.search_tree(xml, 'official-title')['official-title']

    # Check for in-text citation issue
    if len(summary) < 15:
        summary = solve_citation_issue(xml)

    # apply data cleaning steps to the summary
    clean_summary = ps.process(summary, ps.remove_html_tags,
                               ps.lowercase,
                               ps.remove_punctuation,
                               ps.replace_digits,
                               ps.remove_whitespace_chars,
                               ps.shorten_spaces,
                               ps.remove_parenthesis_text,
                               tokenize=False,
                               keep_periods=True
                               )

    return clean_summary


def solve_citation_issue(xml):
    '''
    Collects summaries where an external citation tag in the official-title
    field breaks the additional text post-tag.
    '''
    string_xml = ET.tostring(xml, encoding='unicode')
    regex = re.compile('official-titl.+>(.+)</official-title>', re.DOTALL)
    try:
        summary = re.findall(regex, string_xml)[0]
    except IndexError:
        return ''
    return summary
