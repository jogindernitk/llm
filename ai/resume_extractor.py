import argparse
import json
from pathlib import Path
from typing import Any

from model import ResumeData
from ollama_llm_client import LLMClient, OllamaLLMClient


SYSTEM_PROMPT = """
You are a resume information extraction system.

Extract information only from the supplied resume.
Do not infer or invent information.

Return only a valid JSON object with exactly these fields:

{
  "full_name": string or null,
  "email": string or null,
  "phone": string or null,
  "skills": [string, ...],
  "experience": [string, ...],
  "education": [string, ...]
}

Rules:
- Use null for missing full_name, email, or phone.
- Use an empty array for missing skills, experience, or education.
- Do not include additional fields.
- Do not include Markdown.
- Do not include explanations.
- Return only the JSON object.
""".strip()


def create_user_prompt(resume_text: str) -> str:
    return f"""
Extract the required information from the following resume.

<resume>
{resume_text}
</resume>
""".strip()


def extract_resume(
    resume_text: str,
    llm: LLMClient,
) -> ResumeData:

    user_prompt = create_user_prompt(resume_text)

    response = llm.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        output_schema=ResumeData.model_json_schema()
    )

    print("\n----- RAW LLM RESPONSE -----")
    print(response)
    print("----------------------------\n")

    result = ResumeData.model_validate_json(response)

    return result


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Extract structured information from a resume."
    )

    parser.add_argument(
        "resume",
        type=Path,
        help="Path to a plain-text resume file",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON file",
    )

    args = parser.parse_args()

    # -----------------------------
    # Read resume
    # -----------------------------

    resume_text = args.resume.read_text(
        encoding="utf-8"
    ).strip()

    if not resume_text:
        parser.error("The resume file is empty.")

    # -----------------------------
    # Create LLM provider
    # -----------------------------

    llm = OllamaLLMClient(
        model="qwen3:8b"
    )

    # -----------------------------
    # Extract resume information
    # -----------------------------

    result = extract_resume(
        resume_text=resume_text,
        llm=llm,
    )

    output_json = result.model_dump_json(indent=2)
    # -----------------------------
    # Save result
    # -----------------------------

    output_path = (
        args.output
        or args.resume.with_name(
            f"{args.resume.stem}.parsed.json"
        )
    )
    output_path.write_text(
        output_json + "\n",
        encoding="utf-8",
    )

    print("----- VALIDATED RESULT -----")
    print(output_json)

    print(
        f"\nSaved validated result to: {output_path}"
    )


if __name__ == "__main__":
    main()