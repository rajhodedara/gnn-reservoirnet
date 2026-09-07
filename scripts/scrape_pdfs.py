
import requests, os, urllib3
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

for a in soup.find_all('a', href=True):
    href = a['href']
    if 'ViewPDFList' in href:
        list_url = base_url + href if href.startswith('/') else href
        try:
            r_list = requests.get(list_url, headers=headers, verify=False, timeout=15)
            soup_list = BeautifulSoup(r_list.text, 'html.parser')
            
            for pdf_a in soup_list.find_all('a', href=True):
                pdf_href = pdf_a['href']
                if '.pdf' in pdf_href.lower():
                    if pdf_href.startswith('http'):
                        pdf_url = pdf_href
                    elif pdf_href.startswith('//'):
                        pdf_url = 'https:' + pdf_href
                    elif pdf_href.startswith('/'):
                        pdf_url = base_url + pdf_href
                    else:
                        pdf_url = base_url + '/' + pdf_href.replace('../', '')
                        
                    filename = pdf_url.split('/')[-1].split('?')[0]
                    # skip generic stuff
                    if any(x in filename.lower() for x in ['citizen', 'rti', 'manual', 'act', 'nic']):
                        continue
                        
                    filepath = os.path.join(out_dir, filename)
                    
                    if not os.path.exists(filepath):
                        print('Downloading', filename, 'from', pdf_url)
                        pdf_data = requests.get(pdf_url, headers=headers, verify=False, timeout=30)
                        with open(filepath, 'wb') as f:
                            f.write(pdf_data.content)
        except Exception as inner_e:
            pass

print('Done scraping.')

