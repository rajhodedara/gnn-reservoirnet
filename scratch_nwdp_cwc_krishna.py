import requests
import json
import urllib3
urllib3.disable_warnings()
s = requests.Session()
s.verify = False

url = "https://nwdp.nwic.gov.in/api/3/action/package_search?q=river discharge telemetry hourly cwc&rows=100"
r = s.get(url, timeout=10)
packages = r.json().get('result', {}).get('results', [])
for pkg in packages:
    title = pkg.get("title", "").lower()
    print(f"[{pkg['name']}] {title}")
    for res in pkg.get("resources", []):
        print(f"  -> {res['name']}: {res['id']}")
