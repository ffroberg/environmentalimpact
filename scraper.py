import requests
from bs4 import BeautifulSoup
import pdfplumber
import asyncio
import aiohttp
from io import BytesIO
import sqlite3

async def fetch_pdf(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.read()

async def process_pdf(url):
    chemical_name = url.split('/')[-1].replace('.pdf', '')
    text = ''
    try:
        async with aiohttp.ClientSession() as session:
            content = await fetch_pdf(session, url)
            with BytesIO(content) as stream:
                with pdfplumber.open(stream) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() + '\n'

        return chemical_name, text
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return chemical_name, None

async def main():
    url = 'https://mst.dk/erhverv/sikker-kemi/kemikalier/graensevaerdier-og-kvalitetskriterier/kvalitetskriterier-for-miljoefarlige-forurenende-stoffer-i-vandmiljoeet'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.pdf')]
    
    chemical_data = {}
    tasks = [process_pdf(url) for url in pdf_links]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            print(f"Caught exception: {result}")
            continue
        chemical_name, text = result
        if text:
            #extracted_info = extract_key_information(text, chemical_name)  
            chemical_data[chemical_name] = {'text': text}
        
            #if not extracted_info.get('chemical_name'):
                #extracted_info['chemical_name'] = chemical_name
    
    #print(chemical_data[chemical_name])

    return chemical_data

chemical_data = asyncio.run(main())

# Insert extracted data into the database
conn = sqlite3.connect('chemical_text.db')
c = conn.cursor()
    
for chemical_name, data in chemical_data.items():
    # Insert raw text
    chemical_id = c.lastrowid
    c.execute('''
        INSERT INTO chemical_text (chemical_name, chemical_id, extracted_text)
        VALUES (?, ?, ?)
    ''', (chemical_name, chemical_id, data.get('text', None)))
    
    
conn.commit()
conn.close()
