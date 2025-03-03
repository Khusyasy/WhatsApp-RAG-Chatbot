import os
import shutil
import time
from typing import List

from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = "db"
DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    print(f"Run scraping first or put .md files in {DATA_DIR} directory")
    exit(1)


def create_vector_database():
    embeddings = HuggingFaceEmbeddings(
        model_name="firqaaa/indo-sentence-bert-base",
        # model_kwargs={"device": "cuda"},  # setting to use GPU
    )

    if os.path.exists(PERSIST_DIR):
        print(f"Loading database from persist dir: {PERSIST_DIR}")
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        print("Creating database")

        starttime = time.perf_counter_ns()
        loader = DirectoryLoader(
            DATA_DIR,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
        )
        documents = loader.load()

        # markdown splitter
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=True
        )
        md_splits: List[Document] = []
        for doc in documents:
            splits = markdown_splitter.split_text(doc.page_content)
            md_splits.extend(splits)

        # recursive text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=50,
            separators=["|\n", "  \n", "\n\n"],
        )
        rec_splits = text_splitter.split_documents(md_splits)

        splits = rec_splits

        # buat debugging aja
        with open("debug.txt", "w+", encoding="utf-8", errors="ignore") as d:
            d.write(str(splits))
        print("total data split", len(splits))

        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
        )

        endtime = time.perf_counter_ns()
        print(f"Chroma all finished: {(endtime - starttime)/1_000_000_000} s")

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7},
    )
    return retriever


if __name__ == "__main__":
    if os.path.exists(PERSIST_DIR):
        rebuild = input("Rebuild database? (y/N): ")
        if rebuild.lower() == "y":
            shutil.move(PERSIST_DIR, f"{PERSIST_DIR}_{int(time.time())}")

    retriever = create_vector_database()
    while True:
        message = input("Enter a message (or 'exit' to stop program): ")
        if message == "exit":
            break
        res = retriever.invoke(message)
        print(res)
