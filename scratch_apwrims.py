import requests
import re
import urllib3
urllib3.disable_warnings()
headers = {'User-Agent': 'Mozilla/5.0'}
url = 'https://apwrims.ap.gov.in/'
try:
    r = requests.get(url, verify=False, headers=headers, timeout=10)
    print('Status:', r.status_code)
    scripts = re.findall(r'src="([^"]+\.js[^"]*)"', r.text)
    for s in scripts:
        print(s)
except Exception as e:
    print('Error:', e)
