from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import uvicorn
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import nltk
import socket
import logging
import sys
import traceback

# Import the FLANT5LanguageModel
from flan_t5_model import FLANT5LanguageModel

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api_debug.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Question-Statement Conversion API")

# Get hostname and IP
hostname = "queequeg.local"
ip_address = socket.gethostbyname(hostname)

logger.info(f"Starting server with hostname: {hostname}, IP: {ip_address}")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://{hostname}:3000",
        f"http://{ip_address}:3000",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
MODEL_TYPE = "flan-t5"
MODEL_PATH = "./models"
model = None

class ConversionRequest(BaseModel):
    text: str
    context: Optional[str] = ""
    previous_input: Optional[str] = ""
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_new_tokens: Optional[int] = 200
    min_new_tokens: Optional[int] = 100

class ConversionResponse(BaseModel):
    input_type: str
    input_text: str
    converted_text: str
    
@app.on_event("startup")
async def startup_event():
    global model
    logger.info("Starting up server...")
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        logger.info("NLTK data verified")
        logger.info(f"Initializing model {MODEL_TYPE} from {MODEL_PATH}")
        model = FLANT5LanguageModel(MODEL_TYPE, MODEL_PATH)
        logger.info("Model initialized successfully")
    except LookupError:
        logger.info("Downloading NLTK data...")
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        logger.info("NLTK data downloaded")
        logger.error(f"Error initializing model: {str(e)}")
        logger.error(traceback.format_exc())
        # Don't raise the error, just log it
        model = None
    
    # Initialize the model
    try:
        logger.info(f"Initializing model {MODEL_TYPE} from {MODEL_PATH}")
        model = FLANT5LanguageModel(MODEL_TYPE, MODEL_PATH)
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

@app.post("/convert", response_model=ConversionResponse)
async def convert_text(request: ConversionRequest):
    try:
        if not model:
            logger.error("Model not initialized")
            raise HTTPException(status_code=503, detail="Model not initialized")

        logger.debug(f"Received request: {request}")

        # Generate response using the model
        try:
            response = model.generate_response(
                input_text=request.text,
                context=request.context,
                previous_input=request.previous_input,
                temperature=request.temperature,
                top_p=request.top_p,
                max_new_tokens=request.max_new_tokens,
                min_new_tokens=request.min_new_tokens
            )
            logger.debug(f"Generated response: {response}")
        except Exception as e:
            logger.error(f"Model generation error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

        # Determine if input was question or statement
        try:
            input_type = "question" if model.is_question(request.text) else "statement"
            logger.debug(f"Determined input type: {input_type}")
        except Exception as e:
            logger.error(f"Error determining input type: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Input type error: {str(e)}")

        return ConversionResponse(
            input_type=input_type,
            input_text=request.text,
            converted_text=response
        )

    except Exception as e:
        logger.error(f"Conversion error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    status = {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "hostname": hostname,
        "ip_address": ip_address
    }
    logger.debug(f"Health check: {status}")
    return status

@app.get("/debug")
async def debug_info():
    debug_data = {
        "hostname": socket.gethostname(),
        "ip": socket.gethostbyname(socket.gethostname()),
        "allowed_origins": app.user_middleware[0].middleware.config.allow_origins,
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH
    }
    logger.debug(f"Debug info: {debug_data}")
    return debug_data

@app.get("/debug/model")
async def model_debug():
    model_info = {
        "model_loaded": model is not None,
        "model_type": MODEL_TYPE,
        "model_path": MODEL_PATH,
        "device": str(model.device) if model else None,
        "tokenizer_loaded": model.tokenizer is not None if model else None
    }
    logger.debug(f"Model debug info: {model_info}")
    return model_info

@app.get("/test")
async def test_endpoint():
    """
    Test endpoint to verify basic API functionality
    """
    try:
        test_input = "This is a test statement."
        response = model.generate_response(test_input) if model else "Model not loaded"
        return {
            "status": "ok",
            "test_input": test_input,
            "test_response": response
        }
    except Exception as e:
        logger.error(f"Test endpoint error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
@app.get("/ping")
async def ping():
    logger.info("Ping received")
    return {"status": "ok", "timestamp": str(datetime.datetime.now())}

def run_server(host=ip_address, port=8000):
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        app, 
        host=host, 
        port=port, 
        log_level="debug",
        access_log=True
    )

if __name__ == "__main__":
    logger.info(f"Server running at http://{hostname}:8000 or http://{ip_address}:8000")
    run_server()
