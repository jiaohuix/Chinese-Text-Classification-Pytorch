import gradio as gr
import pandas as pd
import requests
import json

LABELS= [

]

def call_api(texts: list[str]) -> list[dict] | None:
    """
    Calls the FastAPI endpoint to get predictions for a list of texts.

    Args:
        texts: A list of strings to be classified.

    Returns:
        A list of dictionaries, where each dictionary contains the text, label,
        class index, and score. Returns None if the request fails.
    """
    url = "http://localhost:8000/predict/"  # Replace if your API is running elsewhere
    headers = {"Content-Type": "application/json"}
    data = {"texts": texts}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()["predictions"][0]  # Access the list of predictions
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None


def predict_and_display(text):
    """
    Predicts the intention of the input text using the API and displays the results in a Pandas DataFrame.

    Args:
        text: The input text string.

    Returns:
        A Pandas DataFrame containing the label and score of the prediction.
    """
    api_response = call_api([text])

    if api_response:
        prediction = api_response[0]  # Get the first (and only) prediction
        data = {'label': [prediction['label']], 'score': [prediction['score']]}
        df = pd.DataFrame(data)
        return df
    else:
        return pd.DataFrame({'label': ['API 错误'], 'score': [0.0]})  # Return an error DataFrame


if __name__ == "__main__":
    examples = [
        "你好"
    ]
    intentions = "/".join(LABELS)
    
    desc = f"""
        这是一个用于文本分类的演示程序。
        程序将分析您的输入，并预测您的类别，例如：
        {intentions}
        """
    
    iface = gr.Interface(
        fn=predict_and_display,
        inputs=gr.Textbox(lines=2, placeholder="请输入您的问题..."),
        outputs="dataframe",
        title="文本分类",
        description=desc,
        examples=examples
    )
    iface.launch(server_name="0.0.0.0", server_port=8001)
