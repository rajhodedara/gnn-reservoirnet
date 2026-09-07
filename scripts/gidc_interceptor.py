
import asyncio
import json
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        print('[*] Launching browser...')
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Intercept network responses
        async def handle_response(response):
            if 'api' in response.url.lower():
                try:
                    data = await response.json()
                    print(f'\\n[+] Intercepted API: {response.url}')
                    # print snippet
                    print(str(data)[:500])
                except:
                    pass

        page.on('response', handle_response)
        
        print('[*] Navigating to gidcdashboard.in...')
        try:
            await page.goto('https://gidcdashboard.in/', timeout=30000, wait_until='networkidle')
        except Exception as e:
            print('[-] Error during navigation:', e)
            
        print('[*] Waiting for 5 seconds to capture dynamic calls...')
        await page.wait_for_timeout(5000)
        
        await browser.close()
        print('[*] Done.')

if __name__ == '__main__':
    asyncio.run(run())

