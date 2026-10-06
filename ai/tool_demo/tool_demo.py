import json

from ollama import Client


# --------------------------------------------------
# Fake candidate database
# --------------------------------------------------

CANDIDATES = [
    {
        "id": "C001",
        "name": "John Smith",
        "skills": ["Java", "Spring", "AWS"],
    },
    {
        "id": "C002",
        "name": "Sarah Jones",
        "skills": ["Python", "AWS", "Docker"],
    },
    {
        "id": "C003",
        "name": "Michael Brown",
        "skills": ["Python", "FastAPI", "PostgreSQL"],
    },
]


# --------------------------------------------------
# Tool
# --------------------------------------------------

def search_candidates(skill: str) -> list[dict]:
    """Search candidates by technical skill."""

    print(f"\n>>> TOOL EXECUTED: search_candidates({skill!r})")

    return [
        candidate
        for candidate in CANDIDATES
        if any(
            candidate_skill.lower() == skill.lower()
            for candidate_skill in candidate["skills"]
        )
    ]


# --------------------------------------------------
# Ollama
# --------------------------------------------------

client = Client(
    host="http://localhost:11434"
)


# --------------------------------------------------
# Conversation
# --------------------------------------------------

messages = [
    {
        "role": "user",
        "content": "Find candidates with Python experience.",
    }
]


print("Sending request to Qwen...")


response = client.chat(
    model="qwen3:8b",
    messages=messages,
    tools=[search_candidates],
)


print("\n----- FIRST LLM RESPONSE -----")
print(response.message)
print("------------------------------")


# Add Qwen's response to conversation history
messages.append(response.message)


# --------------------------------------------------
# Process tool calls
# --------------------------------------------------

if response.message.tool_calls:

    for tool_call in response.message.tool_calls:

        print("\nLLM requested tool:")
        print(tool_call.function.name)

        print("\nArguments:")
        print(tool_call.function.arguments)

        # IMPORTANT:
        # Python decides which function is actually allowed to execute.

        if tool_call.function.name == "search_candidates":

            arguments = tool_call.function.arguments

            result = search_candidates(
                skill=arguments["skill"]
            )

            print("\nTool result:")
            print(
                json.dumps(
                    result,
                    indent=2,
                )
            )

            # Give tool result back to Qwen
            messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_call.function.name,
                    "content": json.dumps(result),
                }
            )


    # --------------------------------------------------
    # Ask Qwen to produce final answer
    # --------------------------------------------------

    print("\nSending tool result back to Qwen...")


    final_response = client.chat(
        model="qwen3:8b",
        messages=messages,
        tools=[search_candidates],
    )


    print("\n----- FINAL ANSWER -----")
    print(final_response.message.content)
    print("------------------------")

else:

    print(
        "\nQwen did not request a tool."
    )