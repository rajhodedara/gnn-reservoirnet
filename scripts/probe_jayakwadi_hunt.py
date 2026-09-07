import requests
import json

def search_data_gov():
    print("Searching data.gov.in for Daily Reservoir Level and Storage...")
    url = "https://api.data.gov.in/resource/9c659865-ab21-4ffa-a3f9-edbae14f5c86"
    # Actually we don't have an API key, we should try a catalog search.
    # The search API is open:
    search_url = "https://data.gov.in/api/v2/search?title=Daily Reservoir Level"
    try:
        r = requests.get(search_url, timeout=10)
        print(f"data.gov.in search status: {r.status_code}")
        if r.status_code == 200:
            print(json.dumps(r.json(), indent=2)[:1000])
    except Exception as e:
        print(f"data.gov.in error: {e}")

def probe_wris():
    print("\nProbing India-WRIS...")
    url = "https://indiawris.gov.in/wris/assets/js/ReservoirData.json"
    try:
        r = requests.get(url, verify=False, timeout=10)
        print(f"WRIS status: {r.status_code}")
    except Exception as e:
        print(f"WRIS error: {e}")

if __name__ == '__main__':
    search_data_gov()
    probe_wris()
