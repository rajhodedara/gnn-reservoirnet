import json

with open("scratch_nwdp_search.json") as f:
    packages = json.load(f)

print("=== NARMADA ===")
for pkg in packages:
    title = pkg.get("title", "").lower()
    if "narmada" in title or "madhya pradesh" in title or "gujarat" in title:
        for res in pkg.get("resources", []):
            if "discharge" in res.get("name", "").lower():
                print(f"[{pkg['name']}] {title} -> {res['id']}")

print("=== GODAVARI ===")
for pkg in packages:
    title = pkg.get("title", "").lower()
    if "godavari" in title or "maharashtra" in title:
        for res in pkg.get("resources", []):
            if "discharge" in res.get("name", "").lower():
                print(f"[{pkg['name']}] {title} -> {res['id']}")

print("=== KRISHNA ===")
for pkg in packages:
    title = pkg.get("title", "").lower()
    if "krishna" in title or "andhra pradesh" in title or "telangana" in title:
        for res in pkg.get("resources", []):
            if "discharge" in res.get("name", "").lower():
                print(f"[{pkg['name']}] {title} -> {res['id']}")
                
print("=== CAUVERY ===")
for pkg in packages:
    title = pkg.get("title", "").lower()
    if "cauvery" in title or "tamil nadu" in title:
        for res in pkg.get("resources", []):
            if "discharge" in res.get("name", "").lower():
                print(f"[{pkg['name']}] {title} -> {res['id']}")
