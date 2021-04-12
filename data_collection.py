import requests
from requests.exceptions import HTTPError
import json
import xml.etree.ElementTree as ET
import pandas as pd

API_SUMMARIES_INDEX_URL = "https://www.govinfo.gov/bulkdata/json/BILLSUM/"
# FOLDER_NAMES = [str(i) for i in range(113, 118)]
FOLDER_NAMES = list(range(113, 118))
BILL_TYPE_IDS = ['hr', 's']


def collect_summaries():
    '''
    Obtain CRS summaries from govinfo.gov. 
    Does not obtain summaries for Joint, Concurrent, or Simple Resolutions.
    See https://github.com/usgpo/bulk-data/blob/master/Bills-Summary-XML-User-Guide.md
    for more information on bill types.
    '''
    bill_summaries = []
    bill_count = 0

    try:
        for url in [API_SUMMARIES_INDEX_URL + str(i) + '/' for i in FOLDER_NAMES]:
            print("Collecting bill summaries from the " + str(i) + "th Congress")
            for bill_type in BILL_TYPE_IDS:
                summaries_list = http_call(url + bill_type)

                for bill in summaries_list['files']:
                    # If multiple bills exist, then a zip file is provided
                    # Skip for now.
                    if bill['fileExtension'] == 'xml':
                        xml_summary = http_call(bill['link'], return_text=True)
                        bill_summaries.append(
                            parse_xml_summary(
                                xml_summary
                            ))
                    bill_count += 1
                    if bill_count % 100 == 0:
                        print(bill_count, "Bills Counted")

    except HTTPError as e:
        print(e)

    return bill_summaries


def collect_bills_info():
    """
    Collects the names of the files for each bill, as well as a link to
    the text. The summaries don't contain information based on the session
    of Congress that the bill was introduced in. However, the session is 
    used to construct the url that the text of the bill is stored at. Given this,
    we can't construct the URL programatically from the measure ID we collect
    from the summary, unless Congress only had one session (this is only the case
    with the current Congress). As a workaround, we obtain the meta-information
    for each file. We parse the measure ID from each bill, and return a 
    dataframe with the measure ID and the link to the text of the bill. 
    """

    return None


def join_summaries_to_bills(summaries, bills, summaries_col, bills_col):
    """
    Joins the summaries to the link of the bill text based on the measure ID.
    Returns Pandas DataFrame.
    """
    return None


def collect_bill_text(urls):
    """
    Given urls of the bill text, programmatically collects the text from the 
    Govinfo Bulk Data Repository.
    """
    return None


def parse_xml_summary(xml):
    """
    Helper function to parse XML for bill summaries.
    """
    root = ET.fromstring(xml)
    try:
        measure_id = root[0].attrib['measure-id']
        summary = root[0][1].find('summary-text').text
    except Exception as e:
        print(e)
        measure_id = ''
        summary = ''
    return {'id': measure_id, 'summary': summary}


def http_call(url, return_text=False):
    '''
    Calls a url using the requests package and returns 
    json request content. Checks for successful response. If 
    call failed, raises an HTTP Error.
    '''
    headers = {'Accept': 'application/json'}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        if return_text:
            return r.text
        else:
            return r.json()
    else:
        raise HTTPError


if __name__ == "__main__":
    bill_summaries = collect_summaries()
    df = pd.DataFrame(bill_summaries)
    df.to_csv("Summaries.csv", index=False)
