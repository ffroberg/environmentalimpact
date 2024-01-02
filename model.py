'''
@inproceedings{Beltagy2019SciBERT,
  title={SciBERT: Pretrained Language Model for Scientific Text},
  author={Iz Beltagy and Kyle Lo and Arman Cohan},
  year={2019},
  booktitle={EMNLP},
  Eprint={arXiv:1903.10676}
}
'''
from transformers import *
from transformers import AutoTokenizer, AutoModel
import sqlite3
import torch

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

# Connect to the database and fetch the text for a specific chemical
conn = sqlite3.connect('/Users/frejafroberg/Dokumenter (lokal)/environmentalimpact/chemical_text.db')
c = conn.cursor()

c.execute('''
    SELECT extracted_text
    FROM chemical_text
    WHERE chemical_id = 2
''')
result = c.fetchone()
conn.close()

if result:
    chemical_text = result[0]

    # Tokenize the chemical text with appropriate truncation and padding
    # Specify max_length according to your needs, for BERT it's typically 512
    inputs = tokenizer(chemical_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

    # Attempt to run the model and catch any runtime errors
    try:
        #outputs = model(**inputs)
        # Process outputs, e.g., extracting embeddings or features
                
        # Get the embeddings
        with torch.no_grad():  # Ensure no gradients are calculated
            outputs = model(**inputs)

        # Extract the last hidden states
        last_hidden_states = outputs.last_hidden_state  # Shape: (1, seq_len, hidden_size)

        # Mean pooling
        sentence_embedding = torch.mean(last_hidden_states, dim=1)

    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        # Handle or log the error appropriately here

else:
    print("No text found for chemical_id = 2")


