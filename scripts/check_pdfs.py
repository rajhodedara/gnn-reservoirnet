
import fitz # PyMuPDF
import os

pdf_dir = 'data/raw/wris_v2/maharashtra_pdfs'
targets = ['PURNIYANTRAN_BOOK_GMIDC.pdf', 'JID Paithan.pdf']

for t in targets:
    path = os.path.join(pdf_dir, t)
    print(f'\\n--- Inspecting {t} ---')
    try:
        doc = fitz.open(path)
        print(f'Total Pages: {len(doc)}')
        for i in range(min(3, len(doc))):
            text = doc[i].get_text()
            print(f'Page {i+1}:')
            print(text[:200].replace('\\n', ' '))
    except Exception as e:
        print('Error:', e)

