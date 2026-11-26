"""Quick test of /predict endpoint."""
try:
    import requests
except ImportError:
    print("Error: 'requests' library not found. Please install it using 'pip install requests'")
    exit(1)

import glob
import os

# Find a test image
images = glob.glob(r"C:\csp final\CSP\project\Ultrasound_Dataset\Data\Data\train\benign\*.png")
if not images:
    print("No test images found")
    exit(1)

img_path = images[0]
print(f"Testing with: {os.path.basename(img_path)}")

try:
    with open(img_path, "rb") as f:
        resp = requests.post(
            "http://localhost:5000/predict",
            files={"file": ("test.png", f, "image/png")}
        )
    print(f"Status: {resp.status_code}")
    print(f"Content-Type: {resp.headers.get('content-type')}")
    print(f"Content-Length: {len(resp.content)}")
    print(f"Body: {resp.text[:800]}")
except Exception as e:
    print(f"Error: {e}")
