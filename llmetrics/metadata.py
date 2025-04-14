from typing import List
from pydantic import BaseModel

class Metadata(BaseModel):
    model: str
    prompt: str
    prompt_tag: str
    input: str
    input_tag: str
    predicted_label: str
    labels: List[str]