"""
Request schema for POST /api/v1/ask.

Response is NOT a Pydantic model since this endpoint streams
Server-Sent Events rather than a single JSON body — see ask.py's
docstring for the event shapes sent over the stream.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Body of POST /api/v1/ask"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Question for the research agent to answer",
        examples=["How does the transformer architecture handle long sequences?"],
    )