import os
import time
import requests
from typing import Optional
from joblib import Memory
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

cachedir = "./cache"
memory = Memory(cachedir, verbose=0)

@memory.cache
def url_request(
    base_url: str,
    offset=None,
    limit=None,
    api_key: Optional[str] = os.getenv("S2_API_KEY"),
    stop_time: Optional[int] = 0,
):
    """
    Makes an HTTP GET request with optional retries.

    Args:
        base_url (str): The base URL for the API request.
        offset (Optional): Offset parameter for pagination.
        limit (Optional): Limit parameter for pagination.
        api_key (Optional): API key for authentication.
        stop_time (Optional): Time to wait before making the request.

    Returns:
        dict: JSON response from the API.
    """
    time.sleep(stop_time)
    if offset is not None:
        base_url += f"&offset={offset}"
    if limit is not None:
        base_url += f"&limit={limit}"
    if api_key is not None:
        headers = {"x-api-key": api_key}
        return requests.get(base_url, headers=headers).json()
    return requests.get(base_url).json()

# get API key
@retry(
    stop=stop_after_attempt(16),  # Retry up to 5 times
    wait=wait_fixed(2),  # Wait 2 seconds between retries
    retry=retry_if_exception_type(Exception)  # Retry on request exceptions
)
def search_literature_by_plain_text(
    query: str,
    fields: str = "title,abstract,authors,year,venue,publicationVenue,citationCount,openAccessPdf",
    year: str = "2000-",
    batch_size: int = 100,
    max_num: str=1000,
):
    """
    Searches for literature by plain text query.

    Args:
        query (str): Search query.
        fields (str): Fields to include in the results.
        year (str): Year filter for the search.
        batch_size (int): Number of results per batch.

    Returns:
        list: List of retrieved papers.
    """
    base_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields={fields}&year={year}"

    # first check number
    response = url_request(base_url)
    total_paper_num = response["total"]
    print(f"Will retrieve an estimated {response['total']} documents")
    retrieved = 0
    results = []

    for i in range(0, total_paper_num, batch_size):
        response = url_request(base_url, offset=i, limit=batch_size)
        if i + batch_size > max_num:
            print(f"Retrieved a maxium {max_num} papers...")
            break
        retrieved += len(response["data"])
        print(f"Retrieved {retrieved} papers...")
        for paper in response["data"]:
            results.append(paper)

    print(f"Done! Retrieved {retrieved} papers total")
    return results

if __name__ == "__main__":
    results = search_literature_by_plain_text('"Cooperative Threat Reduction Program" AND "partner nations" AND priorities')
    print(len(results))
    if results:
        print(results[0])
