import requests
import json
import urllib3
urllib3.disable_warnings()
s = requests.Session()
s.verify = False

url = "https://nwdp.nwic.gov.in/api/3/action/package_search?q=Pondugala&rows=10"
r = s.get(url, timeout=10)
print("Pondugala:", r.json())

url = "https://nwdp.nwic.gov.in/api/3/action/package_search?q=Wadenepally&rows=10"
r = s.get(url, timeout=10)
print("\nWadenepally:", r.json())

url = "https://nwdp.nwic.gov.in/api/3/action/package_search?q=Wadenapally&rows=10"
r = s.get(url, timeout=10)
print("\nWadenapally:", r.json())

url = "https://nwdp.nwic.gov.in/api/3/action/package_search?q=Wadapally&rows=10"
r = s.get(url, timeout=10)
print("\nWadapally:", r.json())
