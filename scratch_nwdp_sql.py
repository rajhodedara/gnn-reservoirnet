import requests
import urllib3
urllib3.disable_warnings()

s = requests.Session()
s.verify = False

url = "https://nwdp.nwic.gov.in/api/3/action/datastore_search_sql"
sql = "SELECT DISTINCT \"Station\" from \"1b9088b5-d196-4c5d-8780-a888e7e9e86b\""
r = s.get(url, params={"sql": sql}, timeout=10)
print(r.json())
