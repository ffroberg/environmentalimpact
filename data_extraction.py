#%%
from transformers import BertModel, BertTokenizer
import pdfplumber
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import spacy
import pandas as pd
import re
import torch


url = 'https://mst.dk/erhverv/sikker-kemi/kemikalier/graensevaerdier-og-kvalitetskriterier/kvalitetskriterier-for-miljoefarlige-forurenende-stoffer-i-vandmiljoeet'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.pdf')]
chemical_data = {}  # Dictionary to store text for each chemical

for url in pdf_links:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        chemical_name = url.split('/')[-1].replace('.pdf', '')

        with BytesIO(response.content) as stream:
            with pdfplumber.open(stream) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() + '\n'
            chemical_data[chemical_name] = text  # Store text in dictionary under chemical name
        
    except requests.exceptions.HTTPError as e:
        print(f"URL not available: {url}")


chemical_data = {k: v for k, v in list(chemical_data.items())[1:]}

for chemical_name, text in chemical_data.items():
    print(f"Chemical: {chemical_name}")

#%% GPT
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

tokenizerGPT = GPT2Tokenizer.from_pretrained('gpt2')
modelGPT = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizerGPT.pad_token = tokenizerGPT.eos_token
tokenizerGPT.padding_side = 'left'

# Ensure the 'Data' directory exists
os.makedirs('Data', exist_ok=True)

# Define the file path
file_path = os.path.join('Data', "all_chemicals_responses.txt")

with open(file_path, "w") as file:
    for chemical_name, text in chemical_data.items():
        prompt = f"What are the key entities and keywords in this text about {chemical_name}:\n"
        truncated_text = text[:1024 - len(prompt)]  # Adjust truncation as needed
        full_prompt = prompt + truncated_text

        inputs = tokenizerGPT.encode_plus(
            full_prompt, 
            return_tensors='pt', 
            truncation=True, 
            max_length=1024,
            pad_to_max_length=True
        )
        outputs = modelGPT.generate(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_length=500
        )
        response = tokenizerGPT.decode(outputs[0], skip_special_tokens=True)

        # Append each response to the file
        file.write(f"Chemical: {chemical_name}\n")
        file.write(response)
        file.write("\n\n")  # Add some space between entries



#%% BERT
from transformers import BertTokenizer, BertModel
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


for chemical_name, text in chemical_data.items():
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)

# Process the outputs for your specific task
batch_size = 5  # Adjust the batch size based on your machine's capability
all_embeddings = []

# Function to process a batch
def process_batch(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1)

# Create batches and process each
for i in range(0, len(chemical_data), batch_size):
    batch_texts = list(chemical_data.values())[i:i + batch_size]
    batch_embeddings = process_batch(batch_texts)
    all_embeddings.append(batch_embeddings)

# Concatenate all batch embeddings
aggregated_embeddings = torch.cat(all_embeddings, dim=0)

#aggregated_embeddings = torch.mean(outputs.last_hidden_state, dim=1)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10)  # Define the number of clusters
kmeans.fit(aggregated_embeddings.detach().numpy())
clusters = kmeans.labels_

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=0)
reduced_embeddings = tsne.fit_transform(aggregated_embeddings.detach().numpy())

plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters)
plt.colorbar()
plt.show()



#%% checking to see which files i extracted
csv_data = pd.read_csv('Data/chemicals.csv')

csv_chemical_names = csv_data['Navn (link fører til datablad)'].tolist()  # Replace with your column name


from fuzzywuzzy import fuzz

threshold = 75  # Set a threshold for matching, e.g., 80%
matches = []

for csv_name in csv_chemical_names:
    for extracted_name in chemical_data.keys():
        similarity = fuzz.partial_ratio(csv_name, extracted_name)
        if similarity > threshold:
            matches.append((csv_name, extracted_name, similarity))

# Sort matches by similarity score
matches.sort(key=lambda x: x[2], reverse=True)

# Print the matches
for match in matches:
    print(f"CSV Name: {match[0]}, Extracted Name: {match[1]}, Similarity: {match[2]}%")
    print(len(matches))

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertModel.from_pretrained('bert-base-uncased')
#nlp = spacy.load('da_core_news_sm')


