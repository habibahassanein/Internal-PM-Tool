import httpx
from bs4 import BeautifulSoup
import re

async def fetch_and_parse(url: str) -> str:
    """Fetch and parse content from a webpage"""
    try:

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                follow_redirects=True,
                timeout=30.0,
            )
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        text = soup.get_text()

        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > 8000:
            text = text[:8000] + "... [content truncated]"

        return text

    except httpx.TimeoutException:
        return "Error: The request timed out while trying to fetch the webpage."
    except httpx.HTTPError as e:
        return f"Error: Could not access the webpage ({str(e)})"
    except Exception as e:
        return f"Error: An unexpected error occurred while fetching the webpage ({str(e)})"
