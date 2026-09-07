
import fitz
import os

pdf_path = 'data/raw/wris_v2/maharashtra_pdfs/flood_bulletins/PURNIYANTRAN_BOOK_GMIDC.pdf'
out_path = 'scratch/purniyantran_pg1_5.txt'

with open(out_path, 'w', encoding='utf-8') as f:
    try:
        doc = fitz.open(pdf_path)
        f.write(f'Total Pages: {len(doc)}\\n')
        for i in range(min(15, len(doc))):
            text = doc[i].get_text()
            f.write(f'\\n--- Page {i+1} ---\\n')
            f.write(text)
    except Exception as e:
        f.write(f'Error: {e}')

