import requests

base_url = "https://api.semanticscholar.org/datasets/v1/release/"

# Set the release id
release_id = "2025-05-27"

# Make a request to get datasets available the latest release
response = requests.get(base_url + release_id)

# Print the response data
print(response.json())