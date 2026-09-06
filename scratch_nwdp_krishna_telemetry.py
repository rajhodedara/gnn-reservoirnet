import requests

requests.packages.urllib3.disable_warnings()
s = requests.Session()
s.verify = False

r = s.get('https://nwdp.nwic.gov.in/api/3/action/package_search?q=discharge telemetry hourly cwc&rows=1000')
packages = r.json().get('result', {}).get('results', [])

krishna_packages = [p for p in packages if 'krishna' in p.get('title', '').lower() or 'telangana' in p.get('title', '').lower() or 'andhra' in p.get('title', '').lower()]

for pkg in krishna_packages:
    print(pkg['title'])
    for res in pkg.get('resources', []):
        print(f"  {res['name']}: {res['id']}")
