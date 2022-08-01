from fastapi import FastAPI, File, UploadFile
from datetime import datetime
from AI import cat_dog_predict
import os

app = FastAPI()

@app.post("/uploadfiles/")
async def create_files(files: bytes = File(...)):
    filename = datetime.now().strftime('%Y%m%d%H%M%S%f') + ".jpg"
    out_file = open(f"img/{filename}", "wb") # open for [w]riting as [b]inary
    out_file.write( bytes([(file) for file in files]))
    result = cat_dog_predict(f"img/{filename}")
    out_file.close()
    if os.path.isfile(f"img/{filename}"):
        os.remove(f"img/{filename}")

    return result