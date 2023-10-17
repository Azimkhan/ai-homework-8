import io
import logging

import requests
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import numpy as np

app = FastAPI()

# Load your H5 model
model_file = 'best_model.h5'
# model_file = 'dino_dragon_10_0.899.h5'
model = tf.keras.models.load_model(model_file)
print(model.summary())
target_size = (150, 150)


# Define a function to preprocess the image for prediction
def preprocess_image(image):
    img = Image.open(image)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Crop image to square
    # width, height = img.size
    # if width > height:
    #     left = (width - height) / 2
    #     top = 0
    #     right = left + height
    #     bottom = height
    # else:
    #     left = 0
    #     top = (height - width) / 2
    #     right = width
    #     bottom = top + width
    #
    # # Crop the image to the calculated coordinates
    # img = img.crop((left, top, right, bottom))

    # resize image
    img = img.resize(target_size, Image.NEAREST)
    img_arr = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img_arr


# Define the endpoint for image classification
@app.post("/classify")
async def classify_image(file: UploadFile = File(None), image_url: str = Form(...)):
    if not file and not image_url:
        return JSONResponse(content={"error": "Provide an image file or URL."}, status_code=400)

    try:
        if file:
            image = await file.read()
        elif image_url:
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) "}
            response = requests.get(image_url, headers=headers, timeout=10)
            response.raise_for_status()
            image = response.content
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to fetch or read the image: {str(e)}"}, status_code=400)

    processed_image = preprocess_image(io.BytesIO(image))

    # logger = logging.getLogger(__name__)
    # logger.info(f"Image processed. Shape: {processed_image.shape}")
    print(processed_image[0][0])
    prediction = model.predict(np.expand_dims(processed_image, axis=0))

    # Assuming 0 represents "dinosaur" and 1 represents "dragon"
    pred_val = prediction[0][0]
    result = {"class": "dinosaur" if pred_val < 0.5 else "dragon", "confidence": float(pred_val)}

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
