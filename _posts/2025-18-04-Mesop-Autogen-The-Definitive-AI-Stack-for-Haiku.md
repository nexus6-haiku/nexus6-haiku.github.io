---
title: "Mesop + Autogen: The Definitive AI Stack for Haiku?"
date: 2025-04-18T18:30:00-00:00
categories:
  - New Adventures In AI
tags:
  - AI
  - Haiku
  - Python
comments:
  id: 114216883384643825
  #host: mastodon.social
  #user: nexus_6
---

It's an exciting time to be a Haiku user interested in AI. While the rest of the computing world seems to be rushing headlong into increasingly resource-hungry models and frameworks, those of us in the Haiku community have been taking a more measured approach. After all, when your operating system is designed with efficiency and responsiveness as core principles, you tend to look at the AI landscape a bit differently.

After several months of experimenting with various frameworks and approaches, I think I've found a combination that truly works well within Haiku's ecosystem: Mesop for the frontend and Autogen for the backend intelligence. And I'm putting this stack to work in a project I'm calling "Otto" (more on that name choice later).

## The Dependency Dilemma

Anyone who's tried to set up modern AI tools on Haiku knows the struggle: dependencies. So many dependencies. And worse yet, Rust crates that don't recognize Haiku as a supported platform, requiring manual patching and fixes that can consume days of your life.

I've gone through the gauntlet with Streamlit, OpenWebUI, Gradio, and various configurations of LangChain. Each has its merits, but also its own unique set of headaches when it comes to installation and stability on Haiku.

The combination of Mesop and Autogen stands out because they install with minimal fuss. Almost no Rust dependencies means almost no platform compatibility issues to solve.

## What is Mesop?

Mesop is a Python-based web UI framework created by some engineers at Google (though it's not officially supported by Google). It's designed specifically for AI applications, with a clean, functional approach to building interfaces.

What makes it particularly suitable for Haiku?

1. Lightweight dependency chain
2. Python-native (works well with our existing Python packages)
3. Designed with AI applications in mind
4. Clean separation between UI and logic

Let's look at a simplified example from the code I'm using for Otto:

```python
import mesop as me
import mesop.labs as mel
import anthropic
from dotenv import load_dotenv

@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://google.github.io"]
    ),
    path="/",
    title="Mesop Demo Chat",
)
def page():
    mel.chat(transform, title="Anthropic Chat", bot_user="Claude")

def transform(input: str, history: list[mel.ChatMessage]):
    # Get API key from environment variables
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    # Convert message history to Anthropic format
    anthropic_messages = []
    for message in history:
        if message.content.strip():
            role = "user" if message.role == "user" else "assistant"
            anthropic_messages.append({"role": role, "content": message.content})

    # Add the current user message
    anthropic_messages.append({"role": "user", "content": input})

    try:
        # Initialize the Anthropic client
        client = anthropic.Anthropic(api_key=api_key)

        # Stream the response from Anthropic
        with client.messages.stream(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            system="You are a helpful AI assistant named Claude.",
            messages=anthropic_messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        yield f"Error: {str(e)}"
```

The code is remarkably clean. The `@me.page` decorator sets up a web page, and the `mel.chat` function creates a chat interface that calls our `transform` function for each message. The transform function handles converting the message history to the format expected by the Anthropic API, then streams the response back to the UI.

## Enter Autogen: Agents That Actually Work

Microsoft's Autogen framework provides a flexible system for building AI agents. Unlike many other frameworks I've tried, it supports a wide range of models including:

- Ollama/llama.cpp (great for local models)
- OpenAI
- Anthropic
- Gemini (with some caveats I'll address)

But what really sets Autogen apart is its support for the Model Context Protocol (MCP), which allows agents to interact with various tools and services, including the file system.

Here's a simplified example from my Otto project:

```python
import asyncio
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.agents import AssistantAgent

async def main(task: str) -> None:
    # Setup server params for fetching remote content
    fetch_mcp_server = StdioServerParams(command="python3", args=["-m", "mcp_server_fetch"])
    fetch_tools = await mcp_server_tools(fetch_mcp_server)

    # Setup server params for local filesystem access
    fs_mcp_server = StdioServerParams(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/workspace"]
    )
    fs_tools = await mcp_server_tools(fs_mcp_server)

    # Combine all tools
    all_tools = fetch_tools + fs_tools

    # Create an agent that uses Anthropic Claude
    model_client = AnthropicChatCompletionClient(
        model="claude-3-7-sonnet-20250219",
        system_prompt="You are a helpful AI assistant with access to tools."
    )

    # Create the agent
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=all_tools,
        reflect_on_tool_use=False
    )
```

This code sets up an agent with access to tools for fetching content from the web and interacting with the local file system. The agent uses Anthropic's Claude model as its brain, but could just as easily use a local Llama model or OpenAI's GPT.

## The Gemini Caveat

One small hiccup in the Autogen story is Gemini support. The Gemini package doesn't install cleanly on Haiku due to its dependency on Shapely, which insists on building NumPy from scratch rather than using our system-installed version (NumPy 2.2.3 from the depot).

Even after updating NumPy to 2.2.4, the issue persists. It seems the problem lies elsewhere in the dependency chain. I'll be refining the recipe and publishing it soon, but for now, I'm setting Gemini aside and focusing on the models that work flawlessly.

## Why Otto?

So why name my AI assistant project "Otto"? It's a playful nod to the character Otto from the classic 1980s comedy film "Airplane!" For those who might not remember, Otto was the inflatable autopilot who literally took the controls when things got tough. Like that Otto, my AI assistant aims to be a reliable copilot that can help you navigate challenges, even if my version is decidedly less inflatable.



## Is This the Definitive Stack?

After months of experimentation, I believe Mesop + Autogen offers the best combination of ease of installation, flexibility, and power for AI development on Haiku. Is it the definitive stack? Perhaps not for everyone, but it's certainly the most promising I've found so far.

The ability to combine local models via llama.cpp with cloud services like OpenAI and Anthropic provides flexibility that's hard to beat. And with MCP support built into Autogen, the possibilities for creating powerful AI assistants are endless.

I'll be continuing development on Otto, refining the integration between these components and exploring ways to make AI more useful and accessible within Haiku. Watch this space for updates, or check out the code on GitHub to start experimenting yourself.

For now, I'm confident that this combination offers the best path forward for AI on Haiku: lightweight, flexible, and powerful enough to build real applications without fighting the system every step of the way.