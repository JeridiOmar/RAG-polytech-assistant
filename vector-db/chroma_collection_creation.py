import chromadb
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions

# from langchain.text_splitter import SpacyTextSplitter

if __name__ == '__main__':
    #Variable definition
    collection_name = "data-collection-chuncks"
    chroma_persistance_dir = "/home/pred_index_23/chroma-storage/data"
    data_csv = "/home/pred_index_23/scrap-data/scrapv4-cleaned.csv"

    chroma_client = chromadb.PersistentClient(chroma_persistance_dir)
    # sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    #   model_name="distiluse-base-multilingual-cased-v1")sentence-transformers/facebook-dpr-question_encoder-single-nq-base
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="distiluse-base-multilingual-cased-v2")

    loaded = False
    try:
        chroma_client.get_collection(name=collection_name)

        loaded = True
    except:
        loaded = False

    collection = chroma_client.get_or_create_collection(name=collection_name,
                                                        embedding_function=sentence_transformer_ef,
                                                        metadata={"hnsw:space": "cosine"})
    if loaded:
        data = pd.read_csv(data_csv, header=0)
        # text_splitter = SpacyTextSplitter(pipeline='fr_core_news_sm')
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        document_cleaned = []

        documents = data["content"].to_list()
        links = data["link"].to_list()
        counter = 0
        id_track = 0
        for doc in documents:
            if type(doc) is str:
                splitted = text_splitter.split_text(doc)
                metadatas = []
                document_cleaned.extend(splitted)
                for i in range(len(splitted)):
                    metadatas.append({"link": links[counter], "table": "0"})
                collection.add(
                    # documents=documents["content"].to_list(),
                    documents=splitted,
                    metadatas=metadatas,
                    # ids=list(map(str, documents["id"].to_list()))
                    ids=list(map(str, list(range(id_track, len(splitted) + id_track))))
                )
                id_track = id_track + len(splitted)

            counter = counter + 1

    print("storing vectors done")
