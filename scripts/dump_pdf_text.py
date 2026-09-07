
import fitz
import os

pdf_path = 'data/raw/wris_v2/maharashtra_pdfs/JID Paithan.pdf'
out_path = 'scratch/jid_paithan_pg1_5.txt'
os.makedirs('scratch', exist_ok=True)

with open(out_path, 'w', encoding='utf-8') as f:
    try:
        doc = fitz.open(pdf_path)
        f.write(f'Total Pages: {len(doc)}\\n')
        for i in range(min(10, len(doc))):
            text = doc[i].get_text()
            f.write(f'\\n--- Page {i+1} ---\\n')
            f.write(text)
    except Exception as e:
        f.write(f'Error: {e}')

