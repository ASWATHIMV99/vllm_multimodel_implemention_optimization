import requests
import json

# Test text endpoint
def test_text_endpoint():
    url = "http://localhost:5000/text"
    payload = {
        "prompt": "Write a short poem about technology",
        "stream": False
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    print("Text endpoint response:")
    print(response.json())
    print()

# Test image endpoint (you'll need to provide an actual image file)
def test_image_endpoint():
    url = "http://localhost:5000/image"
    payload = {
        'prompt': 'Describe this image in detail'
    }
    # Replace 'path/to/your/image.jpg' with an actual image file path
    files = [
        ('image', ('image.jpg', open('path/to/your/image.jpg', 'rb'), 'image/jpeg'))
    ]
    
    response = requests.post(url, data=payload, files=files)
    print("Image endpoint response:")
    print(response.json())
    print()

# Test audio endpoint (you'll need to provide an actual audio file)
def test_audio_endpoint():
    url = "http://localhost:5000/audio"
    payload = {
        'prompt': 'Describe this audio clip'
    }
    # Replace 'path/to/your/audio.mp3' with an actual audio file path
    files = [
        ('audio', ('audio.mp3', open('path/to/your/audio.mp3', 'rb'), 'audio/mp3'))
    ]
    
    response = requests.post(url, data=payload, files=files)
    print("Audio endpoint response:")
    print(response.json())
    print()

# Test multimodal endpoint (you'll need to provide actual files)
def test_multimodal_endpoint():
    url = "http://localhost:5000/multimodal"
    payload = {
        'text': 'Analyze these media files',
        'prompt': 'Analyze these inputs'
    }
    # Replace with actual file paths
    files = [
        # ('image', ('image.jpg', open('path/to/your/image.jpg', 'rb'), 'image/jpeg')),
        # ('audio', ('audio.mp3', open('path/to/your/audio.mp3', 'rb'), 'audio/mp3'))
    ]
    
    response = requests.post(url, data=payload, files=files)
    print("Multimodal endpoint response:")
    print(response.json())
    print()

if __name__ == "__main__":
    # Make sure the Flask app is running before executing these tests
    test_text_endpoint()
    # test_image_endpoint()  # Uncomment and provide actual image file
    # test_audio_endpoint()  # Uncomment and provide actual audio file
    # test_multimodal_endpoint()  # Uncomment and provide actual files