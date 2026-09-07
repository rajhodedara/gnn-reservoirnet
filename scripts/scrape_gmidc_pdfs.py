
import requests, os, urllib3
from bs4 import BeautifulSoup
urllib3.disable_warnings()

headers = {'User-Agent': 'Mozilla/5.0'}
base_url = 'https://wrd.maharashtra.gov.in'
urls_to_scrape = [
    '/Site/1359/GMIDC',
    '/Site/1377/Flood-Control-Information-Booklet--Upper-Godavari-Basin-and-Girna-Basin'
]

output_dir = 'data/raw/wris_v2/maharashtra_pdfs'
os.makedirs(output_dir, exist_ok=True)

for path in urls_to_scrape:
    print(f'[*] Scraping {base_url}{path}')
    try:
        r = requests.get(base_url + path, headers=headers, verify=False, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        links = soup.find_all('a', href=True)
        pdf_links = [l for l in links if '.pdf' in l['href'].lower()]
        
        for a in pdf_links:
            pdf_url = a['href']
            if not pdf_url.startswith('http'):
                pdf_url = base_url + pdf_url.replace('../', '/')
            
            filename = pdf_url.split('/')[-1].split('?')[0]
            out_path = os.path.join(output_dir, filename)
            
            if not os.path.exists(out_path):
                print(f'  [+] Downloading {filename}...')
                try:
                    pdf_r = requests.get(pdf_url, headers=headers, verify=False, timeout=20)
                    with open(out_path, 'wb') as f:
                        f.write(pdf_r.content)
                except Exception as e:
                    print(f'  [-] Failed to download {filename}: {e}')
            else:
                print(f'  [-] {filename} already exists.')
                
    except Exception as e:
        print(f'[-] Error scraping {path}: {e}')

print('[*] PDF scraping complete.')

