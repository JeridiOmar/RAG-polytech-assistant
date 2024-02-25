# importing pandas as pd
import sys

import pandas as pd
from langchain_community.document_loaders import UnstructuredExcelLoader
import chromadb
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma
from langchain.chains import (
    LLMChain, ConversationalRetrievalChain
)
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.llms import HuggingFaceHub
from langchain.vectorstores import utils as chromautils
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

if __name__ == '__main__':
    # read an excel file and convert
    # into a dataframe object
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    chroma_persistance_dir = "/home/pred_index_23/chroma-storage/data"
    chroma_client = chromadb.PersistentClient(chroma_persistance_dir)
    loader = UnstructuredExcelLoader("/home/pred_index_23/llama-index-data/exportMaquette_MAT_2023_valid_en.xls",
                                     mode="elements")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="distiluse-base-multilingual-cased-v2")

    embedding_function = SentenceTransformerEmbeddings(model_name="distiluse-base-multilingual-cased-v2")

    docs = chromautils.filter_complex_metadata(texts)

    langchain_chroma = Chroma.from_documents(docs, embedding_function)

    llm = HuggingFacePipeline.from_model_id(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 256},
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        langchain_chroma.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True
    )

    # show the dataframe
    chat_history = []
    while True:
        query = input('Prompt: ')
        if query == "exit" or query == "quit" or query == "q":
            print('Exiting')
            sys.exit()
        result = qa_chain({'question': query, 'chat_history': chat_history})
        print('Answer: ' + result['answer'] + '\n')
        chat_history.append((query, result['answer']))
