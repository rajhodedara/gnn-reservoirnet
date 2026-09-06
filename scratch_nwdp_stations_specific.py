import requests
import json
import urllib3
urllib3.disable_warnings()

s = requests.Session()
s.verify = False

def search_station(stn):
    url = f"https://nwdp.nwic.gov.in/api/3/action/datastore_search?resource_id=1b9088b5-d196-4c5d-8780-a888e7e9e86b&filters={{\"Station\":\"{stn}\"}}"
    try:
        r = s.get(url, timeout=10)
        data = r.json()
        print(f"Station {stn} in 1b9088b5 (Telangana): {data.get('result', {}).get('total', 0)} rows")
    except Exception as e:
        print(f"Error {stn}: {e}")

search_station("WADENEPALLY")
search_station("VEERLAPALEM")
search_station("PONDUGALA")
search_station("MADINAPADU")
search_station("DAMARCHERLA")

def search_station_ap(stn):
    # AP manual discharge
    url = f"https://nwdp.nwic.gov.in/api/3/action/datastore_search?resource_id=bc46b1aa-842b-4e52-83f1-e5358171b4a6&filters={{\"Station\":\"{stn}\"}}"
    try:
        r = s.get(url, timeout=10)
        data = r.json()
        print(f"Station {stn} in bc46b1aa (AP): {data.get('result', {}).get('total', 0)} rows")
    except Exception as e:
        print(f"Error AP {stn}: {e}")

search_station_ap("PONDUGALA")
search_station_ap("MADINAPADU")
