# text splitting

from langchain.text_splitter import CharacterTextSplitter

text = '''RAG (retrieval augmented generation) is an advanced NLP model that combines retrieval mechanisms with generative capabilities. RAG aims to improve the accuracy and relevance of its outputs by grounding responses in precise, contextually appropriate data.'''

# Define a text splitter that splits on the '.' character
text_splitter = CharacterTextSplitter(
    separator='.',
    chunk_size=75,  
    chunk_overlap=10  
)

# Split the text using text_splitter
chunks = text_splitter.split_text(text)
print(chunks)
print([len(chunk) for chunk in chunks])


## splitting meaningful chunks

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader = PyPDFLoader("GPR.pdf")
document = loader.load()

# Define a text splitter that splits recursively through the character list
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n', '.', ' ', ''],
    chunk_size=75,  
    chunk_overlap=10  
)

# Split the document using text_splitter
chunks = text_splitter.split_documents(document)
print(chunks)
print([len(chunk.page_content) for chunk in chunks])


## Embedding and storing documents

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# Initialize the OpenAI embedding model
embedding_model = OpenAIEmbeddings(api_key="<OPENAI_API_TOKEN>", model='text-embedding-3-small')

# Create a Chroma vector store and embed the chunks
vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_model
)


from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Generate embeddings
embeddings = model.encode([chunk.page_content for chunk in chunks])

print(embeddings)

from langchain_chroma import Chroma

# Create a Chroma vector store and embed the chunks
vector_store = Chroma.from_documents(
    documents=chunks, 
    embedding=model
)

# creatting the retrieval prompt
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import os
# Set your Hugging Face API token locally and dont share it
huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

prompt = """
Use the only the context provided to answer the following question. If you don't know the answer, reply that you are unsure.
Context: {context}
Question: {question}
"""

# Convert the string into a chat prompt template
prompt_template = ChatPromptTemplate.from_template(prompt)

# define the llm
llm = HuggingFaceEndpoint(repo_id='tiiuae/falcon-7b-instruct', task='text-generation', huggingfacehub_api_token=huggingfacehub_api_token)

# Create an LCEL chain to test the prompt
chain = prompt_template | llm

# Invoke the chain on the inputs provided
print(chain.invoke({"context": "DataCamp's RAG course was created by Meri Nova and James Chapman!", "question": "Who created DataCamp's RAG course?"}))