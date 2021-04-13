import requests
from requests.exceptions import HTTPError
import json
import xml.etree.ElementTree as ET
import pandas as pd

# Use the JSON bulk data links
# More information available at https://github.com/usgpo/bulk-data
SUMMARIES_INDEX_URL = "https://www.govinfo.gov/bulkdata/json/BILLSUM/"
BILLS_INDEX_URL = "https://www.govinfo.gov/bulkdata/json/BILLS"
# IDs for each Congress that summaries are available for
FOLDER_NAMES = list(range(113, 118))
# Only collect bills, no resolutions
BILL_TYPE_IDS = ['hr', 's']


def collect_summaries():
    '''
    Obtain CRS summaries from govinfo.gov. 
    Does not obtain summaries for Joint, Concurrent, or Simple Resolutions.
    See https://github.com/usgpo/bulk-data/blob/master/Bills-Summary-XML-User-Guide.md
    for more information on bill types.
    Returns: list of dictionaries with keys 'id' and 'summary'
    '''
    bill_summaries = []
    bill_count = 0

    try:
        # Get the list of bill types available
        for url in [SUMMARIES_INDEX_URL + str(i) + '/' for i in FOLDER_NAMES]:
            print("Collecting bill summaries from the " +
                  url[-4:-1] + "th Congress")
            # Iterate through each type of bill we want
            for bill_type in BILL_TYPE_IDS:
                # Iterate through each summary
                summaries_list = http_call(url + bill_type)
                for bill in summaries_list['files']:
                    # If multiple bill summaries exist, then a zip file is provided
                    # Skip for now.
                    if bill['fileExtension'] == 'xml':
                        # Obtain the XML file
                        xml_summary = http_call(bill['link'], return_text=True)
                        # Parse XML file to get the ID and summary text
                        bill_summaries.append(
                            parse_xml_summary(
                                xml_summary
                            ))
                    # Track progress in CLI
                    bill_count += 1
                    if bill_count % 100 == 0:
                        print(bill_count, "Summaries Checked")

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
    bills_info = []
    try:
        # Get the list of bill types available
        for url in [BILLS_INDEX_URL + '/' + str(i) + '/' for i in FOLDER_NAMES]:
            print("Collecting bill information from the " +
                  str(url[-4:-1]) + "th Congress")
            # Each Congress (except for current) has 2 sessions
            for session in range(1, 3):
                for bill_type in BILL_TYPE_IDS:

                    bills_list = http_call(
                        url + str(session) + '/' + bill_type)

                    # Iterate through each bill
                    for bill in bills_list['files']:
                        bills_info.append(
                            {
                                'name': bill['name'],
                                'link': bill['link']
                            }
                        )

    except HTTPError as e:
        print(e)
        pass

    return bills_info


def join_summaries_to_bills(summaries, bills, csv=True):
    """
    Joins the summaries to the link of the bill text based on the measure ID.
    If csv is False, assumes Pandas DataFrame
    Returns Pandas DataFrame. 
    """
    if csv:
        summaries = pd.read_csv(summaries)
        bills = pd.read_csv(bills)
    # The IDs are formatted differently
    # This may have to be adjusted, but for now, it appears
    # all of the summary IDs have 'id' prepended to them.
    # And that all of the bill IDs have 'ih' or 'is' appended to them.
    bills['id'] = 'id' + bills.name.str.extract('-(.+)i[sh]\.')

    return pd.merge(summaries, bills, how='left', on='id')


def collect_bill_text(urls, colname, csv=True):
    """
    Given a .csv of urls for the bill text, programmatically collects the text from the 
    Govinfo Bulk Data Repository. This takes a few hours to run, given it is
    making nearly 45,000 calls to the bulk data repository for each file. 
    Some way to parallelize this possible?
    """

    if csv:
        df = pd.read_csv(urls)
    else:
        df = urls

    df = df[df[colname].notnull()].copy()

    df['text'] = df.apply(
        lambda row: http_call(row.link, return_text=True), axis=1
    )

    return df


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


def parse_xml_bill(xml):
    '''
    TO DO
    Unclear right now if we need to retain the XML hierarchy or not. 
    '''
    # root = ET.fromstring(xml)

    return xml


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
    df_summaries = pd.DataFrame(bill_summaries)
    df_summaries.to_csv("Summaries.csv", index=False)

    bills_info = collect_bills_info()
    df_bills_info = pd.DataFrame(bills_info)
    df_bills_info.to_csv("Bills_Info.csv", index=False)

    joined_data = join_summaries_to_bills(
        df_summaries, df_bills_info, csv=False)
    joined_data.to_csv("Joined_Summaries_To_Bills_Info.csv", index=False)

    collect_bill_text(joined_data, 'link', csv=False)
