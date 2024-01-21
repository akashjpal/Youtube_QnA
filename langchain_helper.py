import openai, langchain

from langchain.document_loaders import DirectoryLoader, TextLoader,UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.llms import OpenAI

from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader

def load_and_vectorize_data(url,open_ai_key):
    # Add the youtube transriptor
    loader = YoutubeLoader.from_youtube_url(url)
    documents = loader.load()

    # Set up the RecursiveCharacterTextSplitter, then Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # craete embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=open_ai_key)

    qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="documents-comparative-intelligent-investors",
    )

    return qdrant

def parse_response(response):
    print(response['result'])
    print('\n\nSources:')
    for source_name in response["source_documents"]:
        print(source_name.metadata['source'], "page #:", source_name.metadata['page'])

def get_the_answer(qdrant,query,open_ai_key,k=4):
    llm = OpenAI(temperature=0.7, openai_api_key=open_ai_key)

    # Set up the retriever on the pinecone vectorstore
    # Make sure to set include_metadata = True

    retriever = qdrant.as_retriever(include_metadata=True, metadata_key = 'source')

    # crteating summaries and questions
    summaries = "I want you to act as a youtube assistant.Try to give answer of the Youtube videos in the concise way.If it's like generic questions, then entertain only greeting questions and others just say 'I don't know'"
    question = "Find the answer for this query : {}".format(query)

    # creating template
    template = """
    {summaries}
    {question}
    """

    chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=template,
            input_variables=["summaries", "question"],
            ),
        },
    )

    response = chain(query)
    type(response)
    return response

if __name__ == "__main__":
    print()
    print("|--- Welcome to youtube qna assistant terminal version ---|")
    print()
    url = input("Enter the link of youtube video: ")
    db = load_and_vectorize_data(url,open_ai_key)
    print()
    print("|--- Youtube qna assistant terminal version : loaded your data---|")
    print()
    query = input("Enter your question: ")
    res = get_the_answer(db,query,open_ai_key)
    print()
    print("|--- Youtube qna assistant terminal version : got response---|")
    print()
    print(res['answer'])