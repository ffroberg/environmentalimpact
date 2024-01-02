import re
import sqlite3

def extract_summary_section(text, word_limit=175):
    # Define the regex pattern to capture the summary section heading
    # Simplified pattern to first ensure capturing summaries
    pattern = r"(English Summary|Summary)\s*(A water quality standard|Environmental quality standards)\s*(.*?)(?=\n[A-Z0-9]{2,}|\n[1-9]\.|$)"
   
    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        # Find the position where the heading is found
        start_pos = match.end()
        
        # Capture the next 'word_limit' words after the heading
        subsequent_text = text[start_pos:]
        words = subsequent_text.split()[:word_limit]  # Split the text into words and take the first 'word_limit' words
        
        # Rejoin the words to form the summary section
        summary = ' '.join(words)
        return summary.strip()
    else:
        return "No summary section found"


# Connect to the database
conn = sqlite3.connect('/Users/frejafroberg/Dokumenter (lokal)/environmentalimpact/chemical_text.db')
c = conn.cursor()

# Select all extracted text entries from the database
c.execute('SELECT extracted_text FROM chemical_text')

# Fetch all results
results = c.fetchall()
conn.close()

# Check if any results are returned
if results:
    for result in results:
        # Each 'result' is a tuple, with the first element being the extracted_text
        chemical_text = result[0]
        
        # Extract the summary section from the text
        summary = extract_summary_section(chemical_text)
        print(summary)  # Or further process the summary as needed

else:
    print("No texts found in database")