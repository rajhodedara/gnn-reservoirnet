import urllib.request
import json

def search(query):
    print(f"--- Search: {query} ---")
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8')
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all('a', class_='result__snippet'):
            print(a.text)
    except Exception as e:
        print(f"Error: {e}")

search('site:wrd.maharashtra.gov.in "Jayakwadi" filetype:pdf')
search('site:indiawris.gov.in "Jayakwadi" filetype:csv')
search('site:nca.gov.in "Mandleshwar" filetype:pdf')
search('site:wrd.tn.gov.in "Mettur" filetype:pdf')
search('site:ims.telangana.gov.in "Nagarjuna Sagar"')
search('site:apwrims.ap.gov.in "Nagarjuna Sagar"')
