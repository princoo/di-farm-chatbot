import requests
from pathlib import Path

#  download helper function from Learn pytorch repo

if Path("helper_functions.py").is_file():
  print("helper_function.py already exists, skipping download...")
else:
  print("downloading helper_function.py")

  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)