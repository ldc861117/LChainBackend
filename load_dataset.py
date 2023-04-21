import openai
import pinecone
import json
import os
from tqdm import tqdm  # Import the tqdm library for progress bars
from datasets import load_dataset
import uuid
import logging


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

#### Importing the API keys from a text file
def read_api_keys(file_path):
    api_keys = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            api_keys[key] = value

    return api_keys


file_path = 'Env_variables.txt'
api_keys = read_api_keys(file_path)

OPENAI_API_KEY = api_keys['OPENAI_API_KEY']
PINECONE_API_KEY = api_keys['PINECONE_API_KEY']
PINECONE_ENV = api_keys['PINECONE_ENV']
PINECONE_INDEX_NAME = api_keys['PINECONE_INDEX_NAME']

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

def ada_embedding(content, engine='text-embedding-ada-002'):
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

url = input("Enter a url of huggingface dataset: ")
if url=="":
    url="jamescalam/youtube-transcriptions"
    print("Downloading default dataset...")


# first download the dataset
data = load_dataset(
    url,
    split='train'
)

format_file = ".json"
filename=url.replace('/','_').replace(':','_').replace('.','_')+ format_file
file_path = f'resources/{filename}'
if os.path.exists(file_path):
    print("File already exists!")
    new_data=load_json(file_path)
else:
    new_data = []  # this will store adjusted data
    split_docs = []  # this will store the split documents
    window = 20  # number of sentences to combine
    stride = 2  # number of sentences to 'stride' over, used to create overlap
    print("Adjusting data...")
    for i in tqdm(range(0, len(data), stride)):
        i_end = min(len(data)-1, i+window)
        if data[i]['title'] != data[i_end]['title']:
            # in this case we skip this entry as we have start/end of two videos
            continue
        # create larger text chunk
        text = ' '.join(data[i:i_end]['text'])
        # add to adjusted data list
        new_data.append({
            'start': data[i]['start'],
            'end': data[i_end]['end'],
            'title': data[i]['title'],
            'text': text,
            'id': data[i]['id'],
            'url': data[i]['url'],
            'published': data[i]['published']
        })
    save_json(file_path, new_data)
    print("File saved!")
split_docs=new_data

print("text in the middle of the dataset:")
print(split_docs[int(len(split_docs)/2)])
# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create a Pinecone index
index_name = PINECONE_INDEX_NAME
index = pinecone.Index(index_name)
print("total number of documents:", len(split_docs))
print("Press enter to continue...")
input()
# Embed and upsert paragraphs

# Embed and upsert paragraphs
last_index = 0  # Initialize last index to 0
try:
    with open('upsert_log.txt', 'r') as log_file:
        last_index = int(log_file.readline().strip())
        print(f'Resuming upsert from index {last_index}...')
except FileNotFoundError:
    print('Starting new upsert process...')

for id, doc in tqdm(enumerate(split_docs), total=len(split_docs)):
    if id < last_index:
        continue  # Skip documents that have already been upserted

    text = doc['text']
    doc_id = str(uuid.uuid4())
    if len(text.strip()) > 0:  # Ignore empty paragraphs
        vector = ada_embedding(text)
        upsert_response = index.upsert(vectors=[(doc_id, vector, {"page_content": text})])

        # Log successful upsert operation
        logging.info(f'Upserted document {id} with ID {doc_id}')
        last_index = id

# Write the last index to the log file
with open('upsert_log.txt', 'w') as log_file:
    log_file.write(str(last_index))

print("Done!")