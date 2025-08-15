# import os
# from typing import List
# from fastapi import FastAPI
# from pydantic import BaseModel, Field
# from fastapi.responses import JSONResponse
# from pathlib import Path

# # LangChain bits
# from langchain_openai import ChatOpenAI
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from pydantic import BaseModel as PydanticModel, Field as PydanticField
# from langchain_openai import OpenAIEmbeddings
# from langchain_core.output_parsers import PydanticOutputParser



import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .models import ResolveTicketRequest, ResolveTicketResponse
from .loader import load_docs
from .vectorstore import build_vectorstore
from .rag import run_chain

BASE_DIR = Path(__file__).parent
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR.parent / "data"))

DOCS = load_docs(DATA_DIR)
VS = build_vectorstore(DOCS)

app = FastAPI(title="Knowledge Assistant (Modular)", version="0.1.0")

@app.post("/resolve-ticket", response_model=ResolveTicketResponse)
def resolve_ticket(body: ResolveTicketRequest):
    try:
        return JSONResponse(run_chain(body.ticket_text, VS))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "doc_count": len(DOCS)}


# # ---- Load data from .txt files in data/ folder ----

# # DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
# def load_docs(data_dir_path: str):
#     data_dir = Path(data_dir_path)
#     if not data_dir.exists():
#         raise FileNotFoundError(f"Data directory not found: {data_dir}")
#     docs = {}
#     for file in data_dir.glob("*.txt"):
#         content = file.read_text(encoding="utf-8").strip()
#         if content:
#             docs[file.stem.replace("_", " ").title()] = content
#     if not docs:
#         raise ValueError(f"No valid .txt files found in {data_dir}")
#     return docs

# # Call it with your path
# DOCS = load_docs("C:/Users/saiko/Desktop/Tucows/data")
# print(DOCS)
# # ---- Build FAISS index ----
# def build_vectorstore():
#     texts = []
#     metadatas = []
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     for title, content in DOCS.items():
#         for i, chunk in enumerate(splitter.split_text(content)):
#             label = f"{title} §{i+1}"
#             texts.append(chunk)
#             metadatas.append({"title": title, "label": label})
#     emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
#     return FAISS.from_texts(texts=texts, embedding=emb, metadatas=metadatas)

# VS = build_vectorstore()

# MCP_SYSTEM = (
#     "ROLE: Precise Support Knowledge Assistant. Use only provided context.\n"
#     "TASK: Read the ticket and retrieved context, then output JSON per schema.\n"
#     "SCHEMA: {{\"answer\": string, \"references\": string[], "
#     "\"action_required\": \"none|ask_for_more_info|escalate_to_abuse_team|escalate_to_billing|escalate_to_support\"}}\n"
#     "RULES: Output ONLY valid JSON. If info is missing, set action_required='ask_for_more_info'."
# )


# class MCPResponse(PydanticModel):
#     answer: str = PydanticField(...)
#     references: List[str] = PydanticField(default_factory=list)
#     action_required: str = PydanticField(...)

# PARSER = PydanticOutputParser(pydantic_object=MCPResponse)

# PROMPT = ChatPromptTemplate.from_messages([
#     ("system", MCP_SYSTEM),
#     ("user",
#      "TICKET:\n{ticket}\n\n"
#      "CONTEXT (choose refs from these labels only):\n{labels}\n\n"
#      "SNIPPETS:\n{snippets}\n\n"
#      "Return JSON now.")
# ])

# def run_chain(ticket: str) -> dict:
#     docs = VS.similarity_search(ticket, k=4)
#     labels = list({d.metadata.get("label", d.metadata.get("title", "")) for d in docs})
#     snippets = "\n\n".join([f"[{d.metadata.get('label')}]\n{d.page_content}" for d in docs])

#     llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
#     chain = PROMPT | llm | PARSER
#     result: MCPResponse = chain.invoke({"ticket": ticket, "labels": labels, "snippets": snippets})
#     return result.model_dump()

# # ---- FastAPI ----
# class ResolveTicketRequest(BaseModel):
#     ticket_text: str = Field(..., min_length=1)

# class ResolveTicketResponse(BaseModel):
#     answer: str
#     references: List[str]
#     action_required: str

# app = FastAPI(title="Knowledge Assistant (Simple)", version="0.1.0")

# @app.post("/resolve-ticket", response_model=ResolveTicketResponse)
# def resolve_ticket(body: ResolveTicketRequest):
#     return JSONResponse(run_chain(body.ticket_text))
