from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from openvino.inference_engine import IECore

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

embeddings = download_hugging_face_embeddings()

# Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="medical-bot"

# Loading the index
docsearch=Pinecone.from_existing_index(index_name, embeddings)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

# Save the model to a PyTorch format
llm_model = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

import torch
torch.save(llm_model.model.state_dict(), 'llama_model.pth')

# Convert the model to OpenVINO format using the Model Optimizer
# Run the following command in your terminal:
# mo --input_model llama_model.pth --output_dir openvino_model --model_name llama_model

# Load the OpenVINO model
ie = IECore()
net = ie.read_network('openvino_model/llama_model.xml', 'openvino_model/llama_model.bin')

# Create an OpenVINO executable network
exec_net = ie.load_network(net, 'CPU')

# Modify the llm object to use the OpenVINO model
llm = CTransformers(model=exec_net,
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)