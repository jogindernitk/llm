import argparse
import json
from pathlib import Path
from typing import Any

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


def parse_response(response: str) -> dict[str, Any]:
    """
    Parse the JSON object returned by the LLM.

    We first try parsing the complete response. If the model included
    additional text, we attempt to locate a JSON object inside it.
    """

    try:
        result = json.loads(response)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    for index, character in enumerate(response):
        if character != "{":
            continue

        try:
            result, _ = decoder.raw_decode(response[index:])
        except json.JSONDecodeError:
            continue

        if isinstance(result, dict):
            return result

    raise ValueError(
        "The model response did not contain a valid JSON object."
    )


def validate_result(result: dict[str, Any]) -> None:

    expected = {
        "full_name": (str, type(None)),
        "email": (str, type(None)),
        "phone": (str, type(None)),
        "skills": list,
        "experience": list,
        "education": list,
    }

    # Check that all required keys exist and no unexpected keys exist.
    if set(result.keys()) != set(expected.keys()):
        raise ValueError(
            "The model response is missing required keys "
            "or contains unexpected keys."
        )

    # Validate field types.
    for key, accepted_types in expected.items():

        if not isinstance(result[key], accepted_types):
            raise ValueError(
                f"The value for '{key}' has an invalid type."
            )

        # All list values must contain strings.
        if isinstance(result[key], list):
            if not all(
                isinstance(item, str)
                for item in result[key]
            ):
                raise ValueError(
                    f"Every item in '{key}' must be a string."
                )


def extract_resume(
    resume_text: str,
    llm: LLMClient,
) -> dict[str, Any]:

    user_prompt = create_user_prompt(resume_text)

    response = llm.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    print("\n----- RAW LLM RESPONSE -----")
    print(response)
    print("----------------------------\n")

    result = parse_response(response)

    validate_result(result)

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

    # -----------------------------
    # Save result
    # -----------------------------

    output_path = (
        args.output
        or args.resume.with_name(
            f"{args.resume.stem}.parsed.json"
        )
    )

    output_json = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
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