import math
import json
import requests

from utils import tool
from typing import List
from tavily import TavilyClient
from bs4 import BeautifulSoup


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


BASE_URL = "https://hacker-news.firebaseio.com/v0"


def fetch_item(item_id: int):
    """
    Fetches details of a story by its ID.

    Args:
        item_id (int): The ID of the item to fetch.

    Returns:
        dict: Details of the story.
    """
    url = f"{BASE_URL}/item/{item_id}.json"
    response = requests.get(url)
    return response.json()

    
def fetch_story_ids(story_type: str = "top", limit: int = None):
    """
    Fetches the top story IDs.

    Args:
        story_type: The story type. Defaults to top (`topstories.json`)
        limit: The limit of stories to be fetched.

    Returns:
        List[int]: A list of top story IDs.
    """
    url = f"{BASE_URL}/{story_type}stories.json"
    response = requests.get(url)
    story_ids = response.json()

    if limit:
        story_ids = story_ids[:limit]

    return story_ids


def fetch_text(url: str):
    """
    Fetches the text from a URL (if there's text to be fetched). If it fails,
    it will return an informative message to the LLM.

    Args:
        url: The story URL

    Returns:
        A string representing whether the story text or an informative error (represented as a string)
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:

            html_content = response.content
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text()

            return text_content
        else:
            return f"Unable to fetch content from {url}. Status code: {response.status}"
    except Exception as e:
        return f"An error occurred: {e}"


@tool
def get_hn_stories(limit: int = 5, story_type: str = "top"):
    """
    Fetches the top Hacker News stories based on the provided parameters.

    Args:
        limit (int): The number of top stories to retrieve. Default is 10.
        keywords (List[str]): A list of keywords to filter the top stories.
        story_type (str): The story type

    Returns:
        list[Dict[str, Union[str, int]]]: A list of dictionaries containing
        'story_id', 'title', 'url', and 'score' of the stories.
    """

    if limit:
        story_ids = fetch_story_ids(story_type, limit)
    else:
        story_ids = fetch_story_ids(story_type)

    def fetch_and_filter_stories(story_id):
        return fetch_item(story_id)

    stories = [fetch_and_filter_stories(story_id) for story_id in story_ids]
    fromatted_stories = []

    for story in stories:
        story_info = {
            "title": story.get("title"),
            "url": story.get("url"),
            "score": story.get("score"),
            "story_id": story.get("id"),
        }
        fromatted_stories.append(story_info)

    return fromatted_stories[:limit]


@tool
def get_relevant_comments(story_id: int, limit: int =10):
    """
    Get the most relevant comments for a Hacker News item.

    Args:
        story_id: The ID of the Hacker News item.
        limit: The number of comments to retrieve (default is 10).

    Returns:
        A list of dictionaries, each containing comment details.
    """
    story = fetch_item(story_id)

    if 'kids' not in story:
        return "This item doesn't have comments."

    comment_ids = story['kids']

    comment_details = [fetch_item(cid) for cid in comment_ids]
    comment_details.sort(key=lambda comment: comment.get('score', 0), reverse=True)

    relevant_comments = comment_details[:limit]
    relevant_comments = [comment["text"] for comment in relevant_comments]

    return json.dumps(relevant_comments)


@tool
def get_story_content(story_url: str):
    """
    Gets the content of the story.

    Args:
        story_url: A string representing the story URL

    Returns:
        The content of the story
    """
    return fetch_text(story_url)


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


@tool
def write_str_to_txt(string_data: str, txt_filename: str):
    """
    Writes a string to a txt file.

    This function takes a string and writes it to a text file. If the file already exists,
    it will be overwritten with the new data.

    Args:
        string_data (str): The string containing the data to be written to the file.
        txt_filename (str): The name of the text file to which the data should be written.
    """
    # Write the string data to the text file
    with open(txt_filename, mode='w', encoding='utf-8') as file:
        file.write(string_data)

    print(f"Data successfully written to {txt_filename}")


