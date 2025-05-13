import requests
import json
from typing import Dict, Any

def test_csnli_api(text: str) -> Dict[str, Any]:
    """
    Test the CSNLI API with a given text input.
    
    Args:
        text: Input text to process
        
    Returns:
        API response as a dictionary
    """
    url = "http://localhost:6000/csnli-lid"
    headers = {"Content-Type": "application/json"}
    data = {"text": text}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

def print_response(response: Dict[str, Any]) -> None:
    """Pretty print the API response."""
    if response:
        print("\nAPI Response:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
    else:
        print("No response received")

def main():
    # Test cases
    test_cases = [
        "i thght mosam dfrnt hoga bs fog h",
        "मैं घर जा रहा हूं",
        "I am going home",
        "मैं home जा रहा हूं",
        "kya aap english bol sakte hain?"
    ]
    
    print("Testing CSNLI API...\n")
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Input Text: {text}")
        response = test_csnli_api(text)
        print_response(response)

if __name__ == "__main__":
    main() 