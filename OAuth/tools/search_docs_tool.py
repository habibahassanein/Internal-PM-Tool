import requests
from enum import Enum

class DocVersion(Enum):
    LATEST = "latest"
    CLOUD = "cloud"
    V6_0 = "6.0"
    V5_2 = "5.2"
    V5_1 = "5.1"


def fetch_docs_tool(query: str, max_results: int = 5, version: DocVersion = DocVersion.LATEST.value) -> dict:
    """
    Fetch Documentation pages related to a given query.

    Args:
        query (str): Search query
        max_results (int): Maximum number of results (default: 5)

    Returns:
        dict: Search results with tool metadata
    """

    if version not in DocVersion._value2member_map_:
        return "Invalid version specified. \nAvailable versions are: latest, cloud, 6.0, 5.2, 5.1"
    
    url = "https://scvfwsl9qv-dsn.algolia.net/1/indexes/*/queries"

    headers = {
        "Content-Type": "application/json",
        "x-algolia-application-id": "SCVFWSL9QV",
        "x-algolia-api-key": "b59f28ab9f9e4cc3d80dc8b40e397af3",
    }

    payload = {
        "requests": [
            {
                "indexName": "production",
                "params": (
                    "highlightPreTag=<ais-highlight-0000000000>"
                    "&highlightPostTag=</ais-highlight-0000000000>"
                    f"&filters=locale:\"en\" AND version:\"{version}\""
                    "&attributesToHighlight=[\"title\",\"header\"]"
                    "&attributesToSnippet=[\"text:15\"]"
                    f"&hitsPerPage={max_results}"
                    f"&query={query.lower()}"
                    "&facets=[]"
                    "&tagFilters="
                )
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)

    response.raise_for_status()

    data = []

    for item in response.json().get("results", [])[0].get("hits", []):
        data.append({
            "title": item.get("title"),
            "version": item.get("version"),
            "headers": item.get("header"),
            "text": item.get("text"),
            "url": f"https://docs.incorta.com/{item.get('version')}/{item.get('slug')}"
        })

    return {
        "results": data
    }
    

