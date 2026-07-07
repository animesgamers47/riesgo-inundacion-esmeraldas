import sys
import os

# PythonAnywhere WSGI
# 1. Add this project to the path
path = os.path.dirname(os.path.abspath(__file__))
if path not in sys.path:
    sys.path.append(path)

# 2. Import the app
from app import app as application

# 3. Optional: disable Flask debug
application.debug = False
