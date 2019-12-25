import requests
import pytesseract
from io import BytesIO

response = requests.get('https://i0.wp.com/www.embhack.com/wp-content/uploads/2018/06/hello-world.png')
img = Image.open(BytesIO(response.content))
code = pytesseract.image_to_string(img)
print(code)