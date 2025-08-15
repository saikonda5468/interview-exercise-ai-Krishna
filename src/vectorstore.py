import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def build_vectorstore(docs: dict[str, str]) -> FAISS:
    texts: List[str] = []
    metadatas: List[dict] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for title, content in docs.items():
        for i, chunk in enumerate(splitter.split_text(content)):
            label = f"{title} §{i+1}"
            texts.append(chunk)
            metadatas.append({"title": title, "label": label})
    if not texts:
        raise ValueError("No non-empty chunks produced from your .txt files.")
    emb = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    return FAISS.from_texts(texts=texts, embedding=emb, metadatas=metadatas)