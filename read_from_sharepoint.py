"""
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.client_credential import ClientCredential
import io

# SharePoint site and credentials
site_url = "https://naturalhistorymuseum.sharepoint.com/:f:/r/sites/DCPImageSharing/Shared%20Documents/Entom%20Drawer%20Images?csf=1&web=1&e=54vEeI"
client_id = "8f8d2dc2-5b54-4ec8-a6d0-b5be4ad96c45"
client_secret = "your-client-secret"
tenant_id = "73a29c01-4e78-437f-a0d4-c8553e1960c1"
username = "qiang"
password = "ASD31415926vic"
# Authenticate
# ctx = ClientContext(site_url).with_credentials(ClientCredential(client_id, client_secret))
ctx = ClientContext(site_url).with_user_credentials(username, password)

# Path to the file in SharePoint
file_relative_url = "https://naturalhistorymuseum.sharepoint.com/:i:/r/sites/DCPImageSharing/Shared%20Documents/Entom%20Drawer%20Images/L010225272_final.2500x13222.jpeg?csf=1&web=1&e=t52S3o"  # Change as per your folder structure

# Get file content
file = ctx.web.get_file_by_server_relative_url(file_relative_url)
file_content = file.download(io.BytesIO()).execute_query()

# Save locally or process
with open("downloaded_sample.xlsx", "wb") as local_file:
    local_file.write(file_content.getvalue())

print("File downloaded successfully!")
"""

import requests
from PIL import Image
from io import BytesIO

# Image URL
image_url = "https://naturalhistorymuseum.sharepoint.com/:i:/r/sites/DCPImageSharing/Shared%20Documents/Entom%20Drawer%20Images/L010225272_final.2500x13222.jpeg?csf=1&web=1&e=t52S3o"  # Change as per your folder structure


# Fetch image from URL
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Display the image
image.show()

# Convert to NumPy array (useful for ML models)
import numpy as np
image_array = np.array(image)
print(image_array.shape)  # (height, width, channels)

