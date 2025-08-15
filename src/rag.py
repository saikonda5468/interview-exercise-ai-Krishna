import os
from langchain_openai import ChatOpenAI
from .prompt import PROMPT, PARSER, MCPResponse

def run_chain(ticket: str, vs) -> dict:
    docs = vs.similarity_search(ticket, k=4)
    labels = list({d.metadata.get("label", d.metadata.get("title", "")) for d in docs})
    snippets = "\n\n".join([f"[{d.metadata.get('label')}]\n{d.page_content}" for d in docs])

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your environment or .env file.")

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0, api_key=api_key)
    chain = PROMPT | llm | PARSER
    result: MCPResponse = chain.invoke({"ticket": ticket, "labels": labels, "snippets": snippets})
    return result.model_dump()