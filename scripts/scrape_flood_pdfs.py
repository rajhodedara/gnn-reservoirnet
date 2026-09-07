
import requests, os, urllib3, re
from bs4 import BeautifulSoup
urllib3.disable_warnings()

headers = {'User-Agent': 'Mozilla/5.0'}
base_url = 'https://wrd.maharashtra.gov.in'
start_url = 'https://wrd.maharashtra.gov.in/Site/1377/Flood-Control-Information-Booklet--Upper-Godavari-Basin-and-Girna-Basin'

out_dir = 'data/raw/wris_v2/maharashtra_pdfs/flood_bulletins'
os.makedirs(out_dir, exist_ok=True)

print('Fetching main page...')
r = requests.get(start_url, headers=headers, verify=False, timeout=15)
soup = BeautifulSoup(r.text, 'html.parser')

def download_pdfs_from_page(url):
    print(f'Scanning {url}')
    res = requests.get(url, headers=headers, verify=False, timeout=15)
    page_soup = BeautifulSoup(res.text, 'html.parser')
    found_pdfs = False
    for a in page_soup.find_all('a', href=True):
        href = a['href']
        if 'Upload/PDF/' in href or '.pdf' in href.lower():
            if any(x in href.lower() for x in ['nic', 'rti', 'citizen', 'manual']):
                continue
            
            pdf_url = base_url + href.replace('../../', '/Site/') if href.startswith('../../') else href
            if not pdf_url.startswith('http'):
                pdf_url = base_url + pdf_url if pdf_url.startswith('/') else base_url + '/' + pdf_url
                
            filename = pdf_url.split('/')[-1].split('?')[0]
            filepath = os.path.join(out_dir, filename)
            
            if not os.path.exists(filepath):
                print('  Downloading', filename)
                try:
                    pdf_data = requests.get(pdf_url, headers=headers, verify=False, timeout=30)
                    with open(filepath, 'wb') as f:
                        f.write(pdf_data.content)
                    found_pdfs = True
                except:
                    print('  Failed:', filename)
            else:
                found_pdfs = True
    return page_soup

download_pdfs_from_page(start_url)
print('Done.')

