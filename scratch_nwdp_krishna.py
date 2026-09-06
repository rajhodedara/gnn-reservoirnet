import requests
import urllib3
urllib3.disable_warnings()

s = requests.Session()
s.verify = False

krishna_resources = [
    "156dec2c-93ff-4b0e-a602-ec6b61749bfd",
    "b6091dd4-146d-40f1-bcab-3c258b8054d7",
    "6f9d7e96-d879-4167-9223-4a6fe052b437",
    "e9c33b12-510f-4b37-89c2-372ab5015b08",
    "bc46b1aa-842b-4e52-83f1-e5358171b4a6",
    "6b2cd091-ed44-4cdb-afda-49b92a68a404",
    "72d081d9-941e-4510-a792-bf4892b0b627",
    "733dd9d3-0574-4d08-8234-535de13ed83b",
    "42988dde-0444-4476-8fa7-fca43da40bbf"
]

for r in krishna_resources:
    url = f"https://nwdp.nwic.gov.in/api/3/action/datastore_search?resource_id={r}&limit=5000"
    try:
        resp = s.get(url, timeout=10)
        if resp.ok:
            data = resp.json().get('result', {}).get('records', [])
            if data:
                stns = set(x.get('Station') for x in data)
                print(f"{r}: {stns}")
            else:
                print(f"{r}: No data or different structure")
    except Exception as e:
        print(f"Error {r}: {e}")

