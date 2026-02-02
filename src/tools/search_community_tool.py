import requests
from bs4 import BeautifulSoup
import re
import time

def extract_posts(results: list) -> list:
    """
    Extract posts from the search result.

    Args:
        result (list): Search result list

    Returns:
        list: List of extracted posts
    """
    if results[0].get("data") == ["PLACES"]:
        posts_index = 1
        for idx, item in enumerate(results):
            if item.get("data") == ["POSTS"]:
                posts_index = idx
                break

        return results[posts_index + 1 :]
    
    else:
        return results

    
def extract_post_info(html_string):
    """
    Extracts valuable information from forum post HTML
    
    Args:
        html_string (str): The HTML string containing the post
        
    Returns:
        dict: Parsed post information
    """
    soup = BeautifulSoup(html_string, 'html.parser')
    
    title_link = soup.find('a', class_='lia-autocomplete-message-list-item-link')
    title = title_link.get_text(strip=True) if title_link else None
    url = title_link.get('href') if title_link else None
    
    body_container = soup.find('div', class_='lia-truncated-body-container')
    body_preview = None
    if body_container:
        body_preview = body_container.get_text(strip=True)
        body_preview = re.sub(r'\s+', ' ', body_preview).strip()
    
    comments_count = soup.find('span', class_='lia-message-stats-count')
    comments = int(comments_count.get_text(strip=True)) if comments_count else 0
    
    date_span = soup.find('span', class_='local-date')
    date = date_span.get_text(strip=True) if date_span else None
    
    board_title = soup.find('span', class_='lia-autocomplete-suggestion-board-title')
    category = board_title.get_text(strip=True) if board_title else None
    
    post_type_icon = soup.find('span', class_='lia-img-icon-tkb-board')
    post_type = post_type_icon.get('title') if post_type_icon else None
    
    return {
        'title': title,
        'body_preview': body_preview,
        'comments': comments,
        'date': date,
        'category': category,
        'post_type': post_type,
        'url': f'https://community.incorta.com{url}' if url else None
    }



def fetch_community_tool(query: str, max_results=5) -> dict:
    """
    Fetch community posts related to a given query.

    Args:
        query (str): Search query
        max_results (int): Maximum number of results (default: 10)

    Returns:
        dict: Search results with tool metadata
    """
    url = "https://community.incorta.com/t5/community/page.searchformv32.messagesearchfield.messagesearchfield:autocomplete"

    params = {
        "t:cp": "search/contributions/page",
        "q": query.lower(),
        "limit": max_results,
        "timestamp": str(int(time.time() * 1000)),
        "searchContext": "wdmcw32433|community"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    try:
        data = response.json()
    except Exception:
        return {
            "results": [],
            "error": "Community search returned non-JSON response"
        }

    if data != [] and isinstance(data, list):
        posts_data = extract_posts(data)
        posts_info = [extract_post_info(post.get("data")[0]) for post in posts_data]

        return {
            "results": posts_info
        }
    else:
        return {
            "results": []
        }