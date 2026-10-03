"""The shape of an answer, as data instead of free text.

Two layers of checking, and they do different jobs:
  1. JSON schema (sent to the API): the model must return this exact shape.
     It guarantees structure: the right fields, the right types.
  2. Pydantic validators (run here): rules a schema can't express, which need
     context the model doesn't control, like "source 7 doesn't exist, I only
     gave you 5". If one fails, the error text goes back to the model to fix.
"""
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="One sentence stating a single fact from the sources.")
    sources: list[int] = Field(description="Numbers of the sources that support this sentence.")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("claim text is empty")
        return v.strip()

    @field_validator("sources")
    @classmethod
    def real_sources(cls, v, info: ValidationInfo):
        if not v:
            raise ValueError("every claim needs at least one source number")
        n = (info.context or {}).get("n_sources")
        if n is not None:
            bad = [s for s in v if not 1 <= s <= n]
            if bad:
                raise ValueError(f"source numbers {bad} don't exist; valid numbers are 1 to {n}")
        return sorted(set(v))  # [2, 1, 2] -> [1, 2]


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool = Field(description="False if the sources do not contain the answer.")
    claims: list[Claim] = Field(description="The answer as separate cited sentences. Empty when found is false.")

    @model_validator(mode="after")
    def consistent(self):
        if self.found and not self.claims:
            raise ValueError("found is true but there are no claims")
        if not self.found and self.claims:
            raise ValueError("found is false but claims were given; return an empty list")
        return self


class Grades(BaseModel):
    """The grader's verdict on retrieved chunks (Stage 3)."""
    model_config = ConfigDict(extra="forbid")

    relevant: list[int] = Field(description="Numbers of the sources that contain information answering the question. Empty if none do.")

    @field_validator("relevant")
    @classmethod
    def real_sources(cls, v, info: ValidationInfo):
        n = (info.context or {}).get("n_sources")
        if n is not None and any(not 1 <= s <= n for s in v):
            raise ValueError(f"source numbers must be 1 to {n}")
        return sorted(set(v))


class Rewrite(BaseModel):
    """A new search query when the last one found nothing useful (Stage 3)."""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="A new search query using different words from the earlier ones.")


def response_format(model=Answer):
    """OpenAI structured-outputs wrapper around a Pydantic-generated schema."""
    return {
        "type": "json_schema",
        "json_schema": {"name": model.__name__.lower(), "strict": True, "schema": model.model_json_schema()},
    }
