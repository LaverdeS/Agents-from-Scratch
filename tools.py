import math
import json
import requests

from utils import tool
from typing import List
from tavily import TavilyClient


@tool
def hn_tool(top_n: int):
    """
    Fetch the top stories from Hacker News.

    This function retrieves the top `top_n` stories from Hacker News using the Hacker News API.
    Each story contains the title, URL, score, author, and time of submission. The data is fetched
    from the official Firebase Hacker News API, which returns story details in JSON format.

    Args:
        top_n (int): The number of top stories to retrieve.
    """
    top_stories_url = 'https://hacker-news.firebaseio.com/v0/topstories.json'

    try:
        response = requests.get(top_stories_url)
        response.raise_for_status()  # Check for HTTP errors

        # Get the top story IDs
        top_story_ids = response.json()[:top_n]

        top_stories = []

        # For each story ID, fetch the story details
        for story_id in top_story_ids:
            story_url = f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json'
            story_response = requests.get(story_url)
            story_response.raise_for_status()  # Check for HTTP errors
            story_data = story_response.json()

            # Append the story title and URL (or other relevant info) to the list
            top_stories.append({
                'title': story_data.get('title', 'No title'),
                'url': story_data.get('url', 'No URL available'),
            })

        return json.dumps(top_stories)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return []


@tool
def sum_two_elements(a: int, b: int) -> int:
    """
    Computes the sum of two integers.

    Args:
        a (int): The first integer to be summed.
        b (int): The second integer to be summed.

    Returns:
        int: The sum of `a` and `b`.
    """
    return a + b


@tool
def multiply_two_elements(a: int, b: int) -> int:
    """
    Multiplies two integers.

    Args:
        a (int): The first integer to multiply.
        b (int): The second integer to multiply.

    Returns:
        int: The product of `a` and `b`.
    """
    return a * b


@tool
def compute_log(x: int) -> float | str:
    """
    Computes the logarithm of an integer `x` with an optional base.

    Args:
        x (int): The integer value for which the logarithm is computed. Must be greater than 0.

    Returns:
        float: The logarithm of `x` to the specified `base`.
    """
    if x <= 0:
        return "Logarithm is undefined for values less than or equal to 0."

    return math.log(x)


@tool
def tavily_web_search(query: str) -> List[dict]:
    """
    Performs a web search for each question using tavily API and extracts
    the title and content of different search results.

    Args:
        query (str): The string with the query or topic for the web search

    Returns:
        List (List[dict]): The search results in a list of dictionaries with title and content keys.
    """
    tavily = TavilyClient()
    tavily_response = tavily.search(query)
    response = [
        {
            'title' : result['title'],
            'content' : result['content']
        } for result in tavily_response['results']
    ]
    return response