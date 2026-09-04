import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Intercept responses
        async def handle_response(response):
            if response.request.resource_type in ["fetch", "xhr"]:
                url = response.url
                if "wris" in url.lower() or "api" in url.lower():
                    try:
                        text = await response.text()
                        if "Krishna" in text or "agency" in text.lower() or "tributary" in text.lower() or len(text) < 2000:
                            print(f"\n--- URL: {url} ---")
                            print(text[:1000])
                    except:
                        pass

        page.on("response", handle_response)
        
        print("Navigating to India WRIS...")
        try:
            await page.goto("https://indiawris.gov.in/wris/#/DataDownload", timeout=30000)
            await page.wait_for_timeout(10000) # Wait 10s for dropdowns to load
        except Exception as e:
            print("Error:", e)
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
