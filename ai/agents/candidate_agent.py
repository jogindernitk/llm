import json
from typing import Callable

from ollama import Client


# --------------------------------------------------
# Candidate database
# --------------------------------------------------

CANDIDATES = [
    {
        "id": "C001",
        "name": "John Smith",
        "skills": ["Java", "Spring", "AWS"],
        "experience_years": 10,
        "current_role": "Senior Java Developer",
        "location": "Toronto",
    },
    {
        "id": "C002",
        "name": "Sarah Jones",
        "skills": ["Python", "AWS", "Docker"],
        "experience_years": 8,
        "current_role": "Senior Backend Developer",
        "location": "Mississauga",
    },
    {
        "id": "C003",
        "name": "Michael Brown",
        "skills": ["Python", "FastAPI", "PostgreSQL"],
        "experience_years": 6,
        "current_role": "Python Developer",
        "location": "Toronto",
    },
]


# --------------------------------------------------
# Tools
# --------------------------------------------------

def search_candidates(skill: str) -> list[dict]:
    """Search for candidates who have a specific technical skill."""

    print(
        f"\n>>> Executing search_candidates(skill={skill!r})"
    )

    return [
        {
            "id": candidate["id"],
            "name": candidate["name"],
        }
        for candidate in CANDIDATES
        if any(
            candidate_skill.lower() == skill.lower()
            for candidate_skill in candidate["skills"]
        )
    ]


def get_candidate(candidate_id: str) -> dict:
    """Get full details for a candidate using their candidate ID."""

    print(
        f"\n>>> Executing get_candidate(candidate_id={candidate_id!r})"
    )

    for candidate in CANDIDATES:
        if candidate["id"].lower() == candidate_id.lower():
            return candidate

    return {
        "error": f"Candidate {candidate_id} not found"
    }


# --------------------------------------------------
# Tool registry
# --------------------------------------------------

TOOL_REGISTRY: dict[str, Callable] = {
    "search_candidates": search_candidates,
    "get_candidate": get_candidate,
}


# --------------------------------------------------
# Agent
# --------------------------------------------------

def run_agent(
    user_request: str,
    max_steps: int = 10,
) -> str:

    client = Client(
        host="http://localhost:11434"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a candidate search assistant. "
                "Use the available tools when candidate "
                "information is required. "
                "Do not invent candidate information. "
                "When you have enough information to answer "
                "the user, provide the final answer."
            ),
        },
        {
            "role": "user",
            "content": user_request,
        },
    ]

    available_tools = [
        search_candidates,
        get_candidate,
    ]

    # --------------------------------------------------
    # Agent loop
    # --------------------------------------------------

    for step in range(1, max_steps + 1):

        print(
            f"\n========== AGENT STEP {step} =========="
        )

        response = client.chat(
            model="qwen3:8b",
            messages=messages,
            tools=available_tools,
        )

        assistant_message = response.message

        # Store the assistant response in conversation history.
        messages.append(assistant_message)

        # --------------------------------------------------
        # No tool calls means the model has produced
        # its final answer.
        # --------------------------------------------------

        if not assistant_message.tool_calls:

            print("\nAgent finished.")

            return assistant_message.content

        # --------------------------------------------------
        # Process requested tools
        # --------------------------------------------------

        for tool_call in assistant_message.tool_calls:

            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            print(
                f"\nLLM requested: {tool_name}"
            )

            print(
                f"Arguments: {arguments}"
            )

            # Security/control boundary:
            # only registered tools may execute.

            tool_function = TOOL_REGISTRY.get(
                tool_name
            )

            if tool_function is None:

                result = {
                    "error": (
                        f"Tool '{tool_name}' "
                        "is not allowed."
                    )
                }

            else:

                try:
                    result = tool_function(
                        **arguments
                    )

                except Exception as exc:
                    result = {
                        "error": str(exc)
                    }

            print("\nTool result:")
            print(
                json.dumps(
                    result,
                    indent=2,
                )
            )

            # Give the result back to the LLM.

            messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": json.dumps(result),
                }
            )

    # --------------------------------------------------
    # Safety termination
    # --------------------------------------------------

    raise RuntimeError(
        f"Agent exceeded maximum of {max_steps} steps."
    )


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    request = (
        "Find candidates with Python experience "
        "and give me the full details for candidate C003."
    )

    answer = run_agent(request)

    print("\n========== FINAL ANSWER ==========")
    print(answer)