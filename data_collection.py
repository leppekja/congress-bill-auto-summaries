import requests
from requests.exceptions import HTTPError
import json
import xml.etree.ElementTree as ET

API_SUMMARIES_INDEX_URL = "https://www.govinfo.gov/bulkdata/json/BILLSUM/"
# FOLDER_NAMES = [str(i) for i in range(113, 118)]
FOLDER_NAMES = ['117']
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
        for url in [API_SUMMARIES_INDEX_URL + i + '/' for i in FOLDER_NAMES]:
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
                    if bill_count % 10 == 0:
                        print(bill_count, "Bills Counted")

    except HTTPError as e:
        print(e)

    return bill_summaries


def parse_xml_summary(xml):
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
