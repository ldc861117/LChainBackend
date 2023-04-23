import os
import threading
import uuid
import logging
import json
from tqdm import tqdm
from datasets import load_dataset
from flask import Flask, request, jsonify

import openai
import pinecone

app = Flask(__name__)

# Read API keys from environment variables
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = os.environ['PINECONE_ENV']
PINECONE_INDEX_NAME = os.environ['PINECONE_INDEX_NAME']

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create a Pinecone index
index = pinecone.Index(PINECONE_INDEX_NAME)

def ada_embedding(content, engine='text-embedding-ada-002'):
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

# Dictionary to store the progress of each request
progress = {}

@app.route('/index_dataset', methods=['POST'])
def index_dataset():
    url = request.form.get('url', 'jamescalam/youtube-transcriptions')

    # Generate a unique request ID
    request_id = str(uuid.uuid4())

    # Start a new thread for the upsert process
    upsert_thread = threading.Thread(target=upsert_data, args=(request_id, url))
    upsert_thread.start()

    return jsonify({"status": "success", "message": "Indexing started.", "request_id": request_id})

@app.route('/progress/<request_id>', methods=['GET'])
def get_progress(request_id):
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

def process_data(data):
    new_data = []  # this will store adjusted data
    window = 20  # number of sentences to combine
    stride = 2  # number of sentences to 'stride' over, used to create overlap
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
    return new_data

def upsert_paragraphs(request_id, split_docs):
    total_documents = len(split_docs)
    progress[request_id] = {
        "total": total_documents,
        "processed": 0,
        "percentage": 0
    }

    # Embed and upsert paragraphs
    for id, doc in enumerate(split_docs):
        text = doc['text']
        doc_id = doc['id']
        if len(text.strip()) > 0:  # Ignore empty paragraphs
            vector = ada_embedding(text)
            upsert_response = index.upsert(vectors=[(doc_id, vector, {"page_content": text})])

        # Update the progress dictionary
        progress[request_id]["processed"] = id + 1
        progress[request_id]["percentage"] = round((id + 1) / total_documents * 100, 2)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

