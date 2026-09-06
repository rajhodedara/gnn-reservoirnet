import requests

requests.packages.urllib3.disable_warnings()
s = requests.Session()
s.verify = False

# List of interesting resource IDs from previous search
# I will query 'https://nwdp.nwic.gov.in/api/3/action/datastore_search?resource_id=...'
# But wait, CWC hourly discharge telemetry for Krishna wasn't listed as a separate package.
# Let's search all packages with "discharge telemetry hourly cwc"
r = s.get('https://nwdp.nwic.gov.in/api/3/action/package_search?q=discharge telemetry hourly cwc&rows=1000')
packages = r.json().get('result', {}).get('results', [])

cwc_resources = []
for pkg in packages:
    for res in pkg.get('resources', []):
        cwc_resources.append(res['id'])

print(f"Found {len(cwc_resources)} CWC telemetry resources. Looking for Nagarjuna Sagar/Krishna stations...")

found = False
for res_id in cwc_resources:
    try:
        r2 = s.get(f'https://nwdp.nwic.gov.in/api/3/action/datastore_search?resource_id={res_id}&limit=5', timeout=5)
        data = r2.json()
        if data.get('success'):
            records = data['result']['records']
            if records:
                # print(records[0])
                # Usually there's a 'stationName' or 'Station' or 'site'
                pass
            # Let's search inside the datastore for Nagarjuna Sagar
            r_search = s.get(f'https://nwdp.nwic.gov.in/api/3/action/datastore_search?resource_id={res_id}&q=Nagarjuna', timeout=5)
            if r_search.json().get('success'):
                recs = r_search.json()['result']['records']
                if recs:
                    print(f"Found Nagarjuna in {res_id}!")
                    for rec in recs[:2]: print(rec)
                    found = True
    except Exception as e:
        pass
if not found:
    print("No Nagarjuna Sagar found in CWC telemetry.")
