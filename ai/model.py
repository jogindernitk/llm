from pydantic import BaseModel, Field


class ResumeData(BaseModel):
    full_name: str | None = None
    email: str | None = None
    phone: str | None = None

    skills: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)