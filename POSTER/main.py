from fastapi import FastAPI, UploadFile
from model import model_pipeline
from PIL import Image
import io

app = FastAPI()

@app.post("/ask")
def ask(image: UploadFile):
    content = image.file.read()
    image = Image.open(io.BytesIO(content))
    
    result = model_pipeline(image)
    return{"result":result}
