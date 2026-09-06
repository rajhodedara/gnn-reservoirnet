import requests
import json

resources = [
    # AP / TS
    "156dec2c-93ff-4b0e-a602-ec6b61749bfd",
    "6f9d7e96-d879-4167-9223-4a6fe052b437",
    "733dd9d3-0574-4d08-8234-535de13ed83b",
    # TN
    "79c5b4a5-d20c-486e-a249-9d9227648976",
]

import urllib3
urllib3.disable_warnings()
s = requests.Session()
s.verify = False

for r in resources:
    url = f"https://nwdp.nwic.gov.in/api/3/action/datastore_search?resource_id={r}&limit=5000"
    resp = s.get(url)
    if resp.ok:
        data = resp.json().get('result', {}).get('records', [])
        stations = set(x.get('Station', 'UNKNOWN') for x in data)
        print(f"Resource {r}: {len(data)} rows. Stations: {stations}")
    else:
        print(f"Failed {r}: {resp.status_code}")
