import requests
import json

# Define the API key (replace this with your actual API key)
API_KEY = 'sk-1234123'

# Get the input from the user
input = "Hello world"

# Prepare the request data to be sent to the GPT API
data = {
  'model': 'microsoft/Phi-3-vision-128k-instruct',
  'stream': True,
  'messages': [
    {
      'role': 'user',
      'content': input
    }
  ]
}

# Set the headers for the request
headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer ' + API_KEY
}

# Send the request to the OpenAI API and process each chunk of data as it arrives
response = requests.post('http://localhost:8888/v1/chat/completions', data=json.dumps(data), headers=headers, stream=True)

if response.status_code == 200:
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            print(chunk.decode())
else:
    print("Request failed with status code: ", response.status_code)