import json

with open("scratch_nwdp_search.json") as f:
    packages = json.load(f)

for pkg in packages:
    title = pkg.get("title", "").lower()
    notes = pkg.get("notes", "").lower()
    
    if "discharge" in title or "flow" in title or "level" in title:
        for res in pkg.get("resources", []):
            rtitle = res.get("name", "").lower()
            if "madhya pradesh" in title or "maharashtra" in title or "andhra pradesh" in title or "telangana" in title or "tamil nadu" in title or "karnataka" in title:
                print(f"[{pkg['name']}] {title} -> {res['id']}")
