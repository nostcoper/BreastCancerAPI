import requests
import os

folder_path = "../../Data/Advanced-MRI-Breast-Lesions/AMBL-623"
url = "http://127.0.0.1:5000/upload"

files = [
    ('files', open(os.path.join(folder_path, file), 'rb'))
    for file in os.listdir(folder_path)
    if file.endswith('.dcm') 
]

response = requests.post(url, files=files)

if response.status_code == 200:
    print("Upload successful!")
    print(response.json())
else:
    print(f"Failed to upload files. Status code: {response.status_code}")
    print(response.json())