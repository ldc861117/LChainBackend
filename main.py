from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
#### Countdown
import sys
import time

#### for showing progress bar
from tqdm import tqdm
#### Streaming is supported for ChatOpenAI through callback handling.
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#### Loading Github repo needs to import git-repo
from git import Repo
import os #OS will be needed for the path to the repo or text file
from langchain.document_loaders import GitLoader # GitLoader is used to load documents from a git repo
from langchain.text_splitter import CharacterTextSplitter # CharacterTextSplitter is used to split documents into chunks

#### For embedding and vector store
import pinecone 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
import os
import json
def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

#### Countdown function
def countdown(seconds):
    for i in range(seconds, 0, -1):
        sys.stdout.write(f"\rCounting down: {i} seconds remaining")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\rCountdown complete!                \n")
    sys.stdout.flush()

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


#### Get the url and subpage links
import requests
from bs4 import BeautifulSoup
import html2text
import re
from urllib.parse import urlparse

url = input("Enter a website to extract the URL's from: ")
# replace all '/'':' and '.' in the url to get a string as filename
format_file = ".json"
filename=url.replace('/','_').replace(':','_').replace('.','_')+ format_file

file_path = f'resources/{filename}'
if not os.path.exists(file_path):
    
    main_page = requests.get(url)
    main_soup = BeautifulSoup(main_page.content, "html.parser")

    # Extract the domain of the main URL
    main_domain = urlparse(url).netloc

    # Find all the subpage links
    subpage_links = []
    for link in tqdm(main_soup.find_all("a", href=True)):
        href = link["href"]
        # Extract the domain of the current link
        link_domain = urlparse(href).netloc
        
        if link_domain and link_domain != main_domain:
            # Ignore links with different domains
            continue

        if re.match(r'^http', href) is None:
            subpage_links.append(url + href)
        else:
            subpage_links.append(href)


    # Convert each subpage to text using html2text
    h = html2text.HTML2Text()
    h.ignore_links = True
    all_texts = []


    # Speedup downloads with aiohttp
    import aiohttp
    import asyncio

    async def download_page(link):
        async with aiohttp.ClientSession() as session:
            async with session.get(link, ssl=False) as response:
                content = await response.text()
                return content

    async def convert_subpage_to_text(link, progress_bar):
        try:
            content = await download_page(link)
            subpage_soup = BeautifulSoup(content, "html.parser")
            text = h.handle(str(subpage_soup))
            progress_bar.update(1)
            return text
        except Exception as e:
            print(f"Error processing link {link}: {e}")

    async def download_and_convert_all(subpage_links):
        all_texts = []
        with tqdm(total=len(subpage_links), desc="Downloading and converting pages") as progress_bar:
            tasks = [convert_subpage_to_text(link, progress_bar) for link in subpage_links]
            all_texts = await asyncio.gather(*tasks)
        return all_texts

    # Run the asynchronous download and conversion process
    all_texts = asyncio.run(download_and_convert_all(subpage_links))
    save_json('resources/%s' % filename, all_texts)
else:
    all_texts = load_json('resources/%s' % filename)
    print(f"The file '{file_path}' already exists, skipping download and conversion.")



# Check if all_texts is not empty before trying to access its elements
if not len(all_texts) > 0:
    print("No texts found in all_texts list.")
    exit()


# Initialize the text splitter
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=2000,
    chunk_overlap=0,
    length_function=len,
)
# Split the text content of the all_texts list
split_texts = []
for text in tqdm(all_texts, desc="Splitting texts"):
    split_docs = text_splitter.create_documents(text)
    split_texts.extend(split_docs)

print(f"Number of split documents: {len(split_texts)}")

file_path = 'Env_variables.txt'
api_keys = read_api_keys(file_path)

OPENAI_API_KEY = api_keys['OPENAI_API_KEY']
PINECONE_API_KEY = api_keys['PINECONE_API_KEY']
PINECONE_ENV = api_keys['PINECONE_ENV']
PINECONE_INDEX_NAME = api_keys['PINECONE_INDEX_NAME']

print(f"API loaded from {file_path}")

#### Embedding and vector store


#python_splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=0)
#docs = []
#from langchain.docstore.document import Document

#for doc in tqdm(data, desc="Loading and splitting documents"):
#    split_docs = python_splitter.create_documents(doc.page_content)
#    docs.extend(split_docs)
print("embedding method is OpenAI ada-002\n")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
choice = input("Please input 1 or 2 for 'pinecone' or 'faiss' to continue:")
if choice == "1":
    print("Pinecone selected")
    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV,  # next to api key in console
    )
    index_name = PINECONE_INDEX_NAME
    # Fetch existing IDs from the Pinecone index
    # existing_ids = set(pinecone.list_ids(index_name))

    # Inject embbeded documents into Pinecone
    print("Initializing Pinecone vector store...\n\n")
    docsearch = Pinecone.from_documents(split_texts, embeddings, index_name=index_name)
    #docsearch = Pinecone.from_existing_index(index_name, embeddings)
elif choice == "2":
    print("Faiss selected")
    from langchain.vectorstores import FAISS
    from langchain.document_loaders import TextLoader
    # Initialize FAISS vector store
    format_file = "_index"
    if os.path.exists(filename+format_file):
        print("Loading FAISS index from disk...")
        docsearch = FAISS.load_local(filename+format_file)
    else:
        print("Initializing FAISS vector store...")
        docsearch = FAISS.from_documents(split_texts, embeddings)
        #Save the FAISS index to disk
        #docsearch.save_local(url)
    print("FAISS vector store initialized!")
else:
    print("Please input 1 or 2 for 'pinecone' or 'faiss' to continue:")
    exit()

#### Chat function
chat = ChatOpenAI(temperature=0.9,streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
print("User: ")
question = input()
docs = docsearch.similarity_search(question)
messages = [
    SystemMessage(content="You are a helpful assistant that helps me with AI coding by reading through documents and answering my questions."),
    HumanMessage(content=question + " Please fetch related docs first."),
    AIMessage(content=docs[0].page_content),
    HumanMessage(content="Good. Now please summarize this content and describe how it reflects on the topic of my question."),
]
print("content fetched:\n\n")
print(docs[0].page_content)
print("Generating response...")
print(chat(messages))

