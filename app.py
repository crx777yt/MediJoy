from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import openai
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENROUTER_API_KEY=os.environ.get('OPENROUTER_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medijoy"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Set your correct OpenRouter API key
 # api_key = "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"
api_key = os.getenv("OPENROUTER_API_KEY")
# Configure OpenRouter API (Correct API base)
openai.api_base = "https://openrouter.ai/api/v1"

# Use ChatOpenAI with headers for OpenRouter
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    temperature=0.7,
    openai_api_key=api_key,  # OpenRouter-compatible API key
    openai_api_base="https://openrouter.ai/api/v1"  # Ensure correct API base
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
 app.run(host="0.0.0.0", port= 8080, debug= True)
