import requests
import re
import urllib3
urllib3.disable_warnings()
headers = {'User-Agent': 'Mozilla/5.0'}
url = 'https://apwrims.ap.gov.in/main.115e8209b2bf7054.js'
r = requests.get(url, verify=False, headers=headers)
endpoints = re.findall(r'"(/[^"]*api[^"]*)"', r.text, flags=re.IGNORECASE)
endpoints += re.findall(r'"([^"]*reservoir[^"]*)"', r.text, flags=re.IGNORECASE)
for e in set(endpoints):
    if len(e) < 100: print(e)
