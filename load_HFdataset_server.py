import os
import threading
import uuid
import logging
import json
from tqdm import tqdm
from datasets import load_dataset
from flask import Flask, request, jsonify
import tiktoken
import openai
import pinecone

app = Flask(__name__)

# Read API keys from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create a Pinecone index
index = pinecone.Index(PINECONE_INDEX_NAME)
# Tokenizer for ada-002
encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

def ada_embedding(content, engine='text-embedding-ada-002'):
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

# Dictionary to store the progress of each request
progress = {}

@app.route('/upsert', methods=['POST'])
def index_dataset():
    url = request.form.get('url', 'jamescalam/youtube-transcriptions')

    # Generate a unique request ID
    request_id = str(uuid.uuid4())

    # Start a new thread for the upsert process
    upsert_thread = threading.Thread(target=upsert_data, args=(request_id, url))
    upsert_thread.start()
    print(f"Starting indexing process. Progress dictionary: {progress}")
    return jsonify({"status": "success", "message": "Indexing started.", "request_id": request_id})

@app.route('/progress/<request_id>', methods=['GET'])
def get_progress(request_id):
    print(f"Checking progress for request ID {request_id}. Progress dictionary: {progress}")
    if request_id in progress:
        return jsonify({"status": "success", "progress": progress[request_id]})
    else:
        return jsonify({"status": "error", "message": "Invalid request ID."})

def upsert_data(request_id, url):
    # first download the dataset
    data = load_dataset(
        url,
        split='train'
    )

    # process the data
    new_data = process_data(data)
    split_docs = new_data

    # Embed and upsert paragraphs
    upsert_paragraphs(request_id, split_docs)

def process_data(data, max_tokens=2000, window=20, stride=2):
    new_data = []
    logging.getLogger('processed').setLevel(logging.INFO)
    
    # Check the last processed index from the log file
    last_processed_index = 0
    try:
        with open('resources/processed.log', 'r') as log_file:
            last_log_line = log_file.readlines()[-1]  # Read the last line
            last_processed_index = int(last_log_line.split()[2]) + 1  # Extract the index and add 1
            print(f'Resuming processing from index {last_processed_index}...')
    except FileNotFoundError:
        with open('resources/processed.log', 'x') as log_file:
            print('Starting new processing...')
    logging.getLogger('processed').addHandler(logging.FileHandler('resources/processed.log'))
    # If the processed data file exists, load it
    if os.path.exists('resources/processed_data.json'):
        with open('resources/processed_data.json', 'r') as json_file:
            new_data = json.load(json_file)

    for i in tqdm(range(last_processed_index, len(data), stride)):
        i_end = min(len(data) - 1, i + window)

        # Create larger text chunk
        text = ' '.join(data[i:i_end]['text'])

        # Check if the combined text exceeds the max_tokens
        if len(encoding.encode(text)) > max_tokens:
            continue

        # Collect metadata
        metadata = {}
        for key in data[i].keys():
            if key != 'text':
                metadata[key] = data[i][key]

        # Add to adjusted data list
        new_data.append({**metadata, 'text': text})

        # Save the processed data
        with open('resources/processed_data.json', 'w') as json_file:
            json.dump(new_data, json_file)
        
        logging.getLogger('processed').info(f'Processed document {i} with ID {data[i]["id"]}')

    return new_data


def upsert_paragraphs(request_id, split_docs):
    total_documents = len(split_docs)
    progress[request_id] = {
        "total": total_documents,
        "processed": 0,
        "percentage": 0
    }
    # Check upload log for last index
    last_index = 0  # Initialize last index to 0
    try:
        with open('resources/upsert.log', 'r') as log_file:
            last_log_line = log_file.readlines()[-1]  # Read the last line
            last_index = int(last_log_line.split()[2]) + 1  # Extract the index and add 1
            print(f'Resuming upsert from index {last_index}...')
    except FileNotFoundError:
        with open('resources/upsert.log', 'x') as log_file:
            print('Starting new upsert process...')

    # Embed and upsert paragraphs
    for id, doc in tqdm(enumerate(split_docs), total=len(split_docs)):
        if id < last_index:
            continue

        text = doc['text']
        doc_id = doc['id']
        metadata = {k: v for k, v in doc.items() if k != 'text' and k != 'id'}

        if len(text.strip()) > 0:
            vector = ada_embedding(text)
            upsert_response = index.upsert(vectors=[(doc_id, vector, metadata)])

            # Log successful upsert operation
            logging.info(f'Upserted document {id} with ID {doc_id}')
            last_index = id

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4999)


