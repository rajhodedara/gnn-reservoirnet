
import asyncio
import json
import os
from playwright.async_api import async_playwright

# [N] Neon v4.3 - WRIS API Network Interceptor
# Bypasses IP blocks by riding on the local machines residential IP via a headed browser.

async def run():
    async with async_playwright() as p:
        print("[*] Launching stealth browser (Chromium)...")
        browser = await p.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled"])
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        os.makedirs("data/raw/wris_v2", exist_ok=True)
        dump_file = "data/raw/wris_v2/wris_intercept_log.jsonl"
        
        print(f"[*] Listening for WRIS API responses. Dumping to: {dump_file}")
        
        async def handle_response(response):
            url = response.url.lower()
            # Intercept any API or MapServer calls
            if ("api" in url or "mapserver" in url or "query" in url) and response.status == 200:
                try:
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        data = await response.json()
                        
                        # Only log if it contains actual data payload to avoid clutter
                        if data:
                            with open(dump_file, "a", encoding="utf-8") as f:
                                f.write(json.dumps({"url": response.url, "data": data}) + "\n")
                            print(f"[+] Intercepted JSON Payload from: {response.url.split('?')[0]}")
                except Exception as e:
                    pass

        page.on("response", handle_response)
        
        print("\n[!] Navigating to India-WRIS Reservoir portal...")
        print("[!] -----------------------------------------------------------")
        print("[!] INSTRUCTIONS:")
        print("[!] 1. The browser will open the WRIS portal.")
        print("[!] 2. Use the UI to search/select 'Jayakwadi (Paithan)'.")
        print("[!] 3. Set the date ranges for 2010-2024 (or whatever it allows).")
        print("[!] 4. The script will automatically steal the backend JSON data as it loads.")
        print("[!] 5. Press Ctrl+C in this terminal when you are done.")
        print("[!] -----------------------------------------------------------\n")
        
        try:
            await page.goto("https://indiawris.gov.in/wris/#/reservoir", timeout=60000)
        except Exception as e:
            print(f"[-] Initial load timeout, but browser is still open. Proceed manually: {e}")
            
        # Keep the browser open indefinitely until user kills it
        await page.wait_for_timeout(36000000)

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[*] Interception complete. Shutting down.")

