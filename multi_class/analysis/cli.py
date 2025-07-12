import requests
import json

SERVER_URL = "http://localhost:8000/predict/"


def call_api(texts: list[str]) -> list[dict] | None:
    """
    Calls the FastAPI endpoint to get predictions for a list of texts.

    Args:
        texts: A list of strings to be classified.

    Returns:
        A list of dictionaries, where each dictionary contains the text, label,
        class index, and score. Returns None if the request fails.
    """
    headers = {"Content-Type": "application/json"}
    data = {"texts": texts}

    try:
        response = requests.post(SERVER_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()["predictions"][0]  # Access the list of predictions
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    texts_to_classify = [
        "你好",
    ]

    api_response = call_api(texts_to_classify)

    if api_response:
        print("API Response:")
        for prediction in api_response:
            print(f"Text: {prediction['text']}")
            print(f"  Label: {prediction['label']}")
            print(f"  Class Index: {prediction['class_idx']}")
            print(f"  Score: {prediction['score']:.4f}")  # Format score to 4 decimal places
            print("-" * 20)
    else:
        print("Failed to get predictions from the API.")