import chromadb
import pandas as pd
from langchain.text_splitter import SpacyTextSplitter
from chromadb.utils import embedding_functions

if __name__ == '__main__':
    documents = pd.read_csv("/home/pred_index_23/scrap-data/scrapv4-cleaned.csv", header=0)
    text_splitter = SpacyTextSplitter(pipeline='fr_core_news_sm')
    document_cleaned = []
    documents = documents["content"].to_list()
    for doc in documents:
        document_cleaned.extend(text_splitter.split_text(doc))

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="distiluse-base-multilingual-cased-v1")
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="first-collection",
                                                        embedding_function=sentence_transformer_ef, )
    collection.add(
        # documents=documents["content"].to_list(),
        documents=document_cleaned,
        metadata={"hnsw:space": "cosine"},
        # ids=list(map(str, documents["id"].to_list()))
        ids=list(map(str, list(range(0, len(document_cleaned)))))
    )
    results = collection.query(
        query_texts=["calendrier universitaire 2023"],
        n_results=3,
        where={"link": {"$contains": "polytech.univ-nantes"}}
    )
    print(results)
