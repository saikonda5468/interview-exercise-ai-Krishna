from typing import List
from pydantic import BaseModel, Field

class ResolveTicketRequest(BaseModel):
    ticket_text: str = Field(..., min_length=1)

class ResolveTicketResponse(BaseModel):
    answer: str
    references: List[str]
    action_required: str