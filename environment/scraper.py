import requests

def get_fear_and_greed_index():
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        index = int(data["fear_and_greed"]["score"])
        print(f"Fetched F&G Index from CNN: {index}")
        return index
    except Exception as e:
        print(f"Error fetching CNN F&G Index: {e}")
        return 50
