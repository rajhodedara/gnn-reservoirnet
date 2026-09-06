import requests
import json
import urllib3
urllib3.disable_warnings()

s = requests.Session()
s.verify = False

url = f"https://nwdp.nwic.gov.in/api/3/action/datastore_search?resource_id=1b9088b5-d196-4c5d-8780-a888e7e9e86b&limit=10000"
try:
    r = s.get(url, timeout=10)
    data = r.json().get('result', {}).get('records', [])
    stns = set(x.get('Station') for x in data)
    print(f"Stations in Telangana CWC: {stns}")
except Exception as e:
    print(e)
