import os
from agno.agent import Agent as AgnoAgent
from agno.models.litellm import LiteLLM
from loguru import logger

# Choose a model that's fast and good at summarization, potentially cheaper.
# Example: "openrouter/anthropic/claude-3-haiku-20240307"
# Example: "gpt-3.5-turbo" (via configured LiteLLM)
COMPLEX_MODEL_ID = "openai-gpt-oss-20b-abliterated-uncensored-neo-imatrix"
SIMPLE_MODEL_ID = os.getenv("LLM_MODEL_ID", "qwen3-stargate-sg1-uncensored-abliterated-8b-i1")

# This system message guides the LLM's summarization style.
# The actual content to be summarized will be passed in the .run() method.
SUMMARIZER_SYSTEM_MESSAGE = """You are an expert summarization assistant. \
Your task is to summarize the provided text content that you will receive.
The summary should be comprehensive and between 500-700 words.
It should capture the most critical information relevant for an AI agent that will use this summary for planning its next steps.
Focus on key outcomes, decisions, facts, and figures. Include important details while avoiding conversational fluff.
Output only the summarized text. Do NOT include any preambles, apologies, or self-references like 'Here is the summary:'. Just the summary text itself.
"""

try:
    model_instance = LiteLLM(
        id=COMPLEX_MODEL_ID,
        provider=os.getenv("LLM_PROVIDER", "openai"),
        api_base=os.getenv("OPENAI_API_BASE", None),
    )

    context_summarizer_agno_agent = AgnoAgent(
        name="ContextSummarizer_Agno",
        model=model_instance,
        system_message=SUMMARIZER_SYSTEM_MESSAGE
    )

    logger.info(f"Successfully initialized ContextSummarizer_Agno with model {COMPLEX_MODEL_ID}")
except Exception as e:
    logger.error(f"Failed to initialize ContextSummarizer_Agno: {e}")
    context_summarizer_agno_agent = None # Ensure it's None if init fails

if context_summarizer_agno_agent is None:
    logger.warning("ContextSummarizer_Agno agent could not be initialized. Summarization will fall back to truncation.")
