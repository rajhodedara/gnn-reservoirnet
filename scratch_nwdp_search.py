import requests
import json
import sys
import urllib3
urllib3.disable_warnings()

def search():
    url = "https://nwdp.nwic.gov.in/api/3/action/package_search?q=discharge&rows=300"
    r = requests.get(url, verify=False)
    data = r.json()
    results = data.get("result", {}).get("results", [])
    
    with open("scratch_nwdp_search.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Found {len(results)} packages")

if __name__ == "__main__":
    search()
