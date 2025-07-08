import requests
from zipfile import ZipFile
from io import BytesIO

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
r = requests.get(url)

with ZipFile(BytesIO(r.content)) as zip_ref:
    zip_ref.extractall("data/raw/")
