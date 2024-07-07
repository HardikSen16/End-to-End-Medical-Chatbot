import openvino as ov
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

PINECONE_API_KEY = "2bd5fd9f-2c56-42f9-aa1b-037960262fff"
PINECONE_API_ENV = "gcp-starter"

# Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents

extracted_data = load_pdf("data/")

# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks

text_chunks = text_split(extracted_data)

# Download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = download_hugging_face_embeddings()

# Initialize the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="medical-chatbot"

# Create embeddings for each of the text chunks & store
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

# Convert the LLM model to OpenVINO IR format
llm_model = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama")
ov_model = ov.convert_model(llm_model)

# Save the OpenVINO IR model to a file
ov.save_model(ov_model, 'llm_model.xml', compress_to_fp16=True)

# Load the OpenVINO IR model and compile it
core = ov.Core()
ov_model = core.read_model('llm_model.xml')
compiled_model = ov.compile_model(ov_model, "AUTO")

# Define a prompt template
prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

# Create a Retrieval QA chain with the compiled OpenVINO model
qa = RetrievalQA.from_chain_type(
    llm=compiled_model, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

while True:
    user_input=input(f"Input Prompt:")
    result=qa({"query": user_input})
    print("Response : ", result["result"])