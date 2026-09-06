import requests

def test_wris(reservoir_name):
    api_url = "https://indiawris.gov.in/Dataset/Reservoir"
    params = {
        "stateName": "0", "districtName": "0", "agencyName": "CWC",
        "startdate": "2010-01-01", "enddate": "2024-12-31", "download": "false",
        "page": 0, "size": 100000,
    }
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    
    try:
        resp = s.post(api_url, params=params, timeout=30)
        records = resp.json().get("data", [])
        
        matches = [r for r in records if reservoir_name.lower() in str(r.get("reservoirName", "")).lower()]
        if not matches:
            print(f"{reservoir_name}: NO MATCHES")
            return
            
        inflow_nonzeros = sum(1 for r in matches if r.get("inflow") or r.get("dailyinflow"))
        print(f"{reservoir_name}: Found {len(matches)} records, {inflow_nonzeros} with inflow")
    except Exception as e:
        print(f"{reservoir_name} Error: {e}")

test_wris("Jayakwadi")
test_wris("Nagarjuna Sagar")
test_wris("Sardar Sarovar")
test_wris("Mettur")
