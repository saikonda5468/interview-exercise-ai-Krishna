from typing import List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel as PydanticModel, Field as PydanticField
from langchain_core.output_parsers import PydanticOutputParser

MCP_SYSTEM = (
    "ROLE: Precise Support Knowledge Assistant. Use only provided context.\n"
    "TASK: Read the ticket and retrieved context, then output JSON per schema.\n"
    "SCHEMA: {{\"answer\": string, \"references\": string[], "
    "\"action_required\": \"none|ask_for_more_info|escalate_to_abuse_team|escalate_to_billing|escalate_to_support\"}}\n"
    "RULES: Output ONLY valid JSON. If info is missing, set action_required='ask_for_more_info'."
)

class MCPResponse(PydanticModel):
    answer: str = PydanticField(...)
    references: List[str] = PydanticField(default_factory=list)
    action_required: str = PydanticField(...)

PARSER = PydanticOutputParser(pydantic_object=MCPResponse)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", MCP_SYSTEM),
    ("user",
     "TICKET:\n{ticket}\n\n"
     "CONTEXT (choose refs from these labels only):\n{labels}\n\n"
     "SNIPPETS:\n{snippets}\n\n"
     "Return JSON now.")
])