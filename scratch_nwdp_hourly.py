import json

with open("scratch_nwdp_search.json") as f:
    packages = json.load(f)

for pkg in packages:
    title = pkg.get("title", "").lower()
    if ("telangana" in title or "andhra pradesh" in title) and "hourly" in title and "telemetry" in title:
        print(f"[{pkg['name']}] {title}")
        for res in pkg.get("resources", []):
            print(f"  -> {res['name']}: {res['id']}")
