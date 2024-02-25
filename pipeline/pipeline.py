# importing pandas as pd
import sys
import pandas as pd
# from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader

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
import glob
from enum import Enum
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.evaluation import EmbeddingDistance
from langchain.evaluation import load_evaluator
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
import uuid
import torch
import os
import gc

warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

root_dir = "/home/pred_index_23/llama-index-data"
llm_id = "mistralai/Mistral-7B-Instruct-v0.2"
data_csv = "/home/pred_index_23/scrap-data/scrapv2-cleaned.csv"
chroma_persistance_dir = "/home/pred_index_23/chroma-storage/data"
transformer_model_name = "distiluse-base-multilingual-cased-v2"


class Types(Enum):
    PDF = "pdf"
    CSV = "csv"
    TXT = "txt"


def load_documents(loader, text_splitter):
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    docs = chromautils.filter_complex_metadata(texts)
    return docs


def create_loaders(type, text_splitter):
    files = load_files_per_types(type)
    loaders = []
    for file in files:
        loader = None
        if type == Types.CSV:
            loader = UnstructuredCSVLoader(file, mode="elements")
        if type == Types.PDF:
            loader = PyPDFLoader(file)
        loaders.extend(load_documents(loader, text_splitter))
    return loaders


def load_text(text_splitter):
    data_csv = "/home/pred_index_23/scrap-data/scrapv2-cleaned.csv"
    data = pd.read_csv(data_csv, header=0)
    texts = data["content"].to_list()
    splitted = []
    documents = []
    for text in texts:
        if type(text) is str:
            splitted.extend(text_splitter.split_text(text))

    for text in splitted:
        doc = Document(page_content=text, metadata={"source": "local"})
        documents.append(doc)
    return documents


def get_path_by_type(filename, type):
    return root_dir + '/{type}/{name}'.format(type=type.value, name=filename)


def load_files_per_types(type):
    suffix = '/{type}/*.{type}'.format(type=type.value)
    return glob.glob(root_dir + suffix)


def rag_pipline_test(query, qa_chain):
    chat_history = []
    result = qa_chain({'question': query, 'chat_history': chat_history})
    chat_history.append((query, result['answer']))
    return result['answer']


def load_pipeline_data():
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pdf_docs = create_loaders(Types.PDF, text_splitter)
    csv_docs = create_loaders(Types.CSV, text_splitter)
    txt_docs = load_text(text_splitter)
    docs = []

    docs.extend(pdf_docs)
    docs.extend(csv_docs)
    docs.extend(txt_docs)
    return docs


def launch_chat(qa_chain):
    chat_history = []
    while True:
        query = input('Prompt: ')
        if query == "exit" or query == "quit" or query == "q":
            print('Exiting')
            sys.exit()
        result = qa_chain({'question': query, 'chat_history': chat_history})
        print('Answer: ' + result['answer'] + '\n')
        chat_history.append((query, result['answer']))


def test_efficiency(qa_chain):
    embedding_model = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v2", )
    evaluator = load_evaluator(
        "embedding_distance", distance_metric=EmbeddingDistance.COSINE, embeddings=embedding_model
    )
    dataset = pd.read_csv("/home/pred_index_23/project-test/test_data.csv", on_bad_lines='skip')
    questions = dataset["question"].to_list()
    reference_responses = dataset["reponse"].to_list()
    rag_responses = []
    scores = []
    counter = 0
    for i in range(0, len(questions)):
        print("Treating question {i} / {total}".format(i=i, total=len(questions)))
        question = questions[i]
        ref = reference_responses[i]
        rag_response = rag_pipline_test(question, qa_chain).strip()
        score = evaluator.evaluate_strings(prediction=rag_response, reference=ref)["score"]
        rag_responses.append(rag_response)
        scores.append(score)
    dict = {'question': questions, 'ref': reference_responses, 'response': rag_responses, 'score': scores}
    df = pd.DataFrame(dict)
    df.to_csv('/home/pred_index_23/project-test/test-result.csv', encoding="utf-8", index=False)


def test_efficiency_multipe():
    data_limit = 100

    docs = load_pipeline_data()

    llm_list = ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-hf"]

    embedding_list = ["dangvantuan/sentence-camembert-base", "distiluse-base-multilingual-cased-v2",
                      "Lajavaness/sentence-flaubert-base"]

    prompt_template = r""" 
    Nous avons fourni des informations de contexte ci-dessous. \n
    -informations:\n
    {context}
    \n
    -Compte tenu que de ces informations, veuillez répondre à la question :  {question}
    """
    dataset = pd.read_csv("/home/pred_index_23/project-test/test_data.csv", on_bad_lines='skip')
    questions = dataset["question"].to_list()
    reference_responses = dataset["reponse"].to_list()

    prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)
    with torch.no_grad():
        for k in range(2, 5):
            for model in llm_list:
                print("Treating model {model}".format(model=model))
                for enc in embedding_list:
                    embedding_function = SentenceTransformerEmbeddings(model_name=enc)
                    new_client = chromadb.EphemeralClient()
                    langchain_chroma = Chroma.from_documents(docs, embedding_function, client=new_client,
                                                             collection_name=str(uuid.uuid4()))
                    embedding_model = HuggingFaceEmbeddings(model_name=enc, )
                    evaluator = load_evaluator(
                        "embedding_distance", distance_metric=EmbeddingDistance.COSINE, embeddings=embedding_model
                    )
                    torch.cuda.empty_cache()

                    if model != "google/gemma-7b-it":
                        llm = HuggingFacePipeline.from_model_id(model_id=model,
                                                                model_kwargs={"torch_dtype": torch.float16},
                                                                task="text-generation",
                                                                pipeline_kwargs={"max_new_tokens": 256}, device=0)
                    else:
                        llm = HuggingFacePipeline.from_model_id(
                            model_id=model,
                            model_kwargs={"torch_dtype": torch.float16},
                            task="text-generation",
                            pipeline_kwargs={"max_new_tokens": 256}, device=0
                        )
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm,
                        langchain_chroma.as_retriever(search_kwargs={'k': k}),
                        return_source_documents=False,
                        combine_docs_chain_kwargs={"prompt": prompt}
                    )
                    rag_responses = []
                    scores = []
                    times = []
                    print("Treating encoding {model}".format(model=enc))
                    counter = 0
                    size_of_pass = data_limit
                    for i in range(0, len(questions)):
                        if counter > data_limit - 1:
                            break
                        print("Treating question {i} / {total}".format(i=i, total=len(questions)))
                        question = questions[i]
                        ref = reference_responses[i]
                        start_time = time.time()
                        # rag_response = rag_pipline_test(question, qa_chain).strip()
                        gc.collect()
                        torch.cuda.empty_cache()
                        try:
                            rag_response = qa_chain({'question': question, 'chat_history': []})['answer'].strip()
                        except:
                            counter = counter + 1
                            size_of_pass = size_of_pass - 1
                            print("Skipped question {i} / {total}".format(i=i, total=len(questions)))
                            continue
                        torch.cuda.empty_cache()
                        gc.collect()
                        end_time = time.time()
                        score = evaluator.evaluate_strings(prediction=rag_response, reference=ref)["score"]
                        rag_responses.append(rag_response)
                        scores.append(score)
                        times.append(end_time - start_time)
                        del question
                        del ref
                        del start_time
                        del rag_response
                        del end_time
                        del score
                        counter = counter + 1
                        torch.cuda.empty_cache()
                    try:

                        dict = {'question': questions[:size_of_pass], 'ref': reference_responses[:size_of_pass],
                                'response': rag_responses, 'score': scores,
                                'time': times, 'llm': model, 'transformer': enc, 'k': k}
                        df = pd.DataFrame(dict)
                    except:
                        size = min(len(rag_responses), len(scores), len(times), len(questions),
                                   len(reference_responses))
                        dict = {'question': questions[:size], 'ref': reference_responses[:size],
                                'response': rag_responses[:size], 'score': scores[:size],
                                'time': times[:size], 'llm': model, 'transformer': enc, 'k': k}
                        df = pd.DataFrame(dict)

                    df.to_csv('/home/pred_index_23/benchmarks/test-result-{uuid}.csv'.format(uuid=str(uuid.uuid4())),
                              encoding="utf-8", index=False)
                    del llm
                    del qa_chain
                    del scores
                    del times
                    del rag_responses
                    del dict
                    # except:
                    #     continue


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Dependencies init
        chroma_client = chromadb.PersistentClient(chroma_persistance_dir)

        embedding_function = SentenceTransformerEmbeddings(model_name=transformer_model_name)

        docs = load_pipeline_data()

        langchain_chroma = Chroma.from_documents(docs, embedding_function)

        llm = HuggingFacePipeline.from_model_id(
            model_id=llm_id,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 256},
        )

        prompt_template = r"""
        Nous avons fourni des informations de contexte ci-dessous. \n
        -informations:\n
        {context}
        \n
        -Compte tenu que de ces informations, veuillez répondre à la question :  {question}
        """
        prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            langchain_chroma.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        launch_chat(qa_chain)


        # To launch benchmarks instead of Chat bot version

        #test_efficiency_multipe()
