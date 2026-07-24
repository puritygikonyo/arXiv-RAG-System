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
    invite_token: str | None = Field(
        default=None,
        description=(
            "Invite token identifying who's asking — sent by the Gradio "
            "web UI after login. Used to enforce revocation and daily "
            "usage limits."
        ),
    )
    invite_token: str | None = Field(
        default=None,
        description=(
            "Invite token identifying who's asking — sent by the Gradio "
            "web UI after login. Used to enforce revocation and daily "
            "usage limits. Optional so /docs testing without a token "
            "still works, but requests without one are rejected by "
            "check_invite_allowed() if invite enforcement is required."
        ),
    )