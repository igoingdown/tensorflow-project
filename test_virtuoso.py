# -*- coding:utf-8 -*-

import sys
import urllib, json
import requests
from bs4 import BeautifulSoup
import re

# Setting global variables
query_url = 'http://10.108.112.28:8890/sparql/'
PREFIX = "<http://rdf.freebase.com/ns/>"


# HTTP URL is constructed accordingly with JSON query results format in mind.
def sparql_query():
    http_result = """
    <table class="sparql" border="1">
    <tr>
    <th>name</th>
    </tr>
    <tr>
    <td>"Ocean Township"@en</td>
    </tr>
    <tr>
    <td>"اوشن، نیوجرسی"@fa</td>
    </tr>
    <tr>
    <td>"Municipio de Ocean"@es</td>
    </tr>
    </table>
    """
    print "ccccc"
    # print http_result
    name_pattern = re.compile("<td>(.*?)@en</td>", re.S)
    alias = name_pattern.findall(http_result)
    # print len(alias)
    # print alias[0]
    s = BeautifulSoup(http_result, "lxml")
    name_list = s.find_all("table")
    print name_list
    print len(name_list)
    for name in name_list:
        print name.get_text()
    ts = s.table.find_all("th")
    for t in ts:
        print t.get_text()

    print len(s.find_all(text="en"))
    print len(s.find_all(border ="1"))

    soup_2 = BeautifulSoup(open("page1.html", "r").read(), "lxml")
    print len(soup_2.find_all(id="smLandingCanada"))
    print soup_2.find_all(id="smLandingCanada")[0].get_text()
    print len(soup_2.find_all(type="text/javascript"))
    for script in soup_2.find_all(type="text/javascript"):
        print script.get_text()

    print len(soup_2.find_all({"class": "landing_us_uk_can"}))

if __name__ == '__main__':
    sparql_query()
