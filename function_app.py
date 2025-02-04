import azure.functions as func  
import datetime  
import json  
import logging  
import requests  
from PIL import Image  
from io import BytesIO  
from sentence_transformers import SentenceTransformer  
  
# Define the Azure Function app  
app = func.FunctionApp()  
  
# Route for the HTTP trigger function  
@app.route(route="img_embed", auth_level=func.AuthLevel.ANONYMOUS)  
def img_embed(req: func.HttpRequest) -> func.HttpResponse:  
    logging.info('Python HTTP trigger function processed a request.')  
  
    # Parse the request for image URLs  
    try:  
        req_body = req.get_json()  
    except ValueError:  
        return func.HttpResponse(  
            "Invalid JSON in request body.",  
            status_code=400  
        )  
      
    image_urls = req_body.get('image_urls')  
    if not image_urls or not isinstance(image_urls, list):  
        return func.HttpResponse(  
            "Please provide a list of image URLs in the request body.",  
            status_code=400  
        )  
      
    try:  
        # Generate image embeddings  
        embeddings_result = image_embeddings(image_urls)  
        return func.HttpResponse(  
            json.dumps(embeddings_result),  
            status_code=200,  
            mimetype="application/json"  
        )  
    except Exception as e:  
        logging.error(f"Error generating image embeddings: {e}")  
        return func.HttpResponse(  
            f"An error occurred: {str(e)}",  
            status_code=500  
        )  
  
# Function to generate image embeddings  
def image_embeddings(image_urls):  
    # Initialize the model  
    model = SentenceTransformer('jinaai/jina-clip-v1', trust_remote_code=True)  
      
    # Load images from URLs  
    try:  
        images = []  
        for url in image_urls:  
            response = requests.get(url)  
            response.raise_for_status()  # Raise an error for bad responses  
            image = Image.open(BytesIO(response.content))  
            images.append(image)  
    except Exception as e:  
        raise ValueError(f"Error loading images: {e}")  
      
    # Generate embeddings  
    image_embeddings = model.encode(images,use_fast=True)  
  
    # Return embeddings as a JSON-serializable structure  
    return {  
        'image-embeddings': image_embeddings.tolist()  # Convert numpy arrays to lists  
    }
