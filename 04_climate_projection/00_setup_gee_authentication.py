"""
============================================================================
SETUP: Google Earth Engine Authentication

One-time setup before running extract_nexgddp_milan.py. Authenticates the
local Python environment with the Earth Engine Cloud project.

Cloud project: nima-21-11-2025
Account:       nimamohammadi.mhmd@gmail.com

Run this script once. After successful authentication, subsequent scripts
that use Earth Engine will initialise without prompting.
============================================================================
"""

import sys
import subprocess

try:
    import ee
except ImportError:
    print("Installing earthengine-api...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "earthengine-api"])
    import ee

CLOUD_PROJECT = "nima-21-11-2025"

print("Authenticating with Google Earth Engine...")
print("A browser window will open. Sign in with nimamohammadi.mhmd@gmail.com")
print("and authorise the Earth Engine application.")
print()

try:
    ee.Authenticate()
    ee.Initialize(project=CLOUD_PROJECT)
    print(f"\nSuccess. Authenticated to Cloud project: {CLOUD_PROJECT}")

    # Sanity check
    img = ee.Image("USGS/SRTMGL1_003")
    info = img.getInfo()
    print(f"Sanity check passed: SRTM image bands = {info.get('bands', [])[0].get('id', 'unknown')}")
except Exception as e:
    print(f"\nERROR: {e}")
    print("If authentication failed, ensure the Cloud project is set up at:")
    print(f"  https://console.cloud.google.com/earth-engine/projects/{CLOUD_PROJECT}")
    sys.exit(1)
