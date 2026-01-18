import requests

def fetch_support_tool(query: str, max_results: int = 5) -> dict:
    """
    Fetch Support pages related to a given query.

    Args:
        query (str): Search query
        max_results (int): Maximum number of results (default: 10)

    Returns:
        dict: Search results with tool metadata
    """
    url = "https://support.incorta.com/hc/api/internal/instant_search.json"
    params = {
        "query": query.lower(),
        "locale": "en-us",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    items = data.get("results", [])[:max_results]

    results = []
    for item in items:
        page_info = {
            "title": item.get("stripped_title", ""),
            "url": f"https://support.incorta.com{item.get('url', '')}",
            "date": item.get("date", ""),
            "type": item.get("type", ""),
            "breadcrumbs": item.get("breadcrumbs", []),
        }
        results.append(page_info)



    return {
        "results": results
    }
