
import asyncio
import json
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        async def handle_response(response):
            if response.request.resource_type in ['xhr', 'fetch']:
                print(f'\\n[+] API: {response.url}')
                try:
                    data = await response.json()
                    print(str(data)[:200])
                except:
                    pass

        page.on('response', handle_response)
        
        try:
            await page.goto('https://gidcdashboard.in/', timeout=30000, wait_until='networkidle')
            await page.wait_for_timeout(5000)
        except Exception as e:
            print('[-] Error:', e)
            
        await browser.close()

if __name__ == '__main__':
    asyncio.run(run())

