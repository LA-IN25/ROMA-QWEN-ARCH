"""
Web Search Tool for AgnoAgent Integration

This module provides a web_search function that connects to a local SearXNG instance.
"""

import requests
import os
from typing import List, Any
from loguru import logger

def web_search(query: str) -> str:
    """
    Performs a web search using a local SearXNG instance and returns formatted results.
    
    Args:
        query: The search query to execute
        
    Returns:
        The search results as a formatted string, or an error message.
    """
    # Get the SearXNG URL from environment variables, with a fallback.
    searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8080")
    
    params = {
        'q': query,
        'format': 'json',
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    logger.info(f"Executing web_search with query: '{query}' against SearXNG at {searxng_url}")
    
    try:
        response = requests.get(searxng_url, params=params, headers=headers, timeout=15)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        data = response.json()
        results = data.get('results', [])
        
        if not results:
            logger.warning(f"No results found for query: '{query}'")
            return "No search results found."
        
        # Format the top 5 results into a clean string for the LLM
        formatted_results = []
        for i, res in enumerate(results[:5], 1):
            title = res.get('title', 'No Title')
            snippet = res.get('content', 'No Snippet Available.')
            url = res.get('url', 'No URL Available.')
            formatted_results.append(
                f"Result {i}:\nTitle: {title}\nURL: {url}\nSnippet: {snippet}\n---"
            )
            
        return "\n".join(formatted_results)

    except requests.exceptions.RequestException as e:
        error_msg = f"Search failed: Could not connect to SearXNG at {searxng_url}. Is it running? Error: {e}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during search: {e}"
        logger.error(error_msg)
        return error_msg

def clean_tools(tools: List[Any]) -> List[Any]:
    """
    Clean tool objects by removing problematic attributes. (No changes needed here)
    """
    def clean_function_obj(func_obj):
        for attr in ['requires_confirmation', 'external_execution']:
            if hasattr(func_obj, attr):
                delattr(func_obj, attr)
        return func_obj

    cleaned = []
    for tool in tools:
        if hasattr(tool, "functions") and isinstance(tool.functions, dict):
            for name, fn in tool.functions.items():
                tool.functions[name] = clean_function_obj(fn)
        if hasattr(tool, "function"):
            tool.function = clean_function_obj(tool.function)
        if callable(tool):
            tool = clean_function_obj(tool)
        cleaned.append(tool)
    return cleaned

__all__ = ['web_search', 'clean_tools']

