from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Literal, Optional
import logging

# Import the core functions from existing modules
from multimodel_therapist import get_therapist_response
from multimodel_friend import get_friend_response
from fer import capture_emotion_from_video

# ------------------------------------------------------------------------------
# App Initialization and Configuration
# ------------------------------------------------------------------------------

# Create FastAPI app instance
app = FastAPI(title="Multi-Model Chat API", version="2.0.0")

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi_model_chat_api")


# ------------------------------------------------------------------------------
# Request Models (Pydantic)
# ------------------------------------------------------------------------------

class TherapistRequest(BaseModel):
    """
    Request schema for /therapist endpoint.
    """
    query: str = Field(..., description="User's input message or question")
    stress: float = Field(..., ge=0.0, description="Stress level as a float (e.g., 0.0 to 1.0 or any chosen scale)")
    mood: str = Field(..., description="Self-reported mood (string label)")
    fatigue: float = Field(..., ge=0.0, description="Fatigue level as a float")
    recovery: float = Field(..., ge=0.0, description="Recovery level as a float")
    fer_mood: str = Field(..., description="Facial emotion recognition mood label")


class FriendRequest(BaseModel):
    """
    Request schema for /friend endpoint.
    """
    query: str = Field(..., description="User's input message or question")
    mode: str = Field(..., description="Friend reply mode (e.g., caring, chill, flirty, funny, deep, hype, real talk)")
    friend_name: str = Field(..., description="Name of the friend persona")
    fer_emotion: Optional[str] = Field(default="neutral", description="Detected emotion from FER (optional)")


class FERCaptureRequest(BaseModel):
    """
    Request schema for /fer/capture endpoint.
    """
    duration_seconds: int = Field(default=5, ge=1, le=30, description="Duration to capture video in seconds")


class FriendWithFERRequest(BaseModel):
    """
    Request schema for /friend/with-fer endpoint - captures FER first then responds.
    """
    query: str = Field(..., description="User's input message or question")
    mode: str = Field(..., description="Friend reply mode")
    friend_name: str = Field(..., description="Name of the friend persona")
    fer_duration: int = Field(default=5, ge=1, le=30, description="Duration for FER capture in seconds")


# ------------------------------------------------------------------------------
# Health Check
# ------------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Lightweight health probe for liveness/readiness checks.
    """
    return {"status": "ok"}


# ------------------------------------------------------------------------------
# FER Endpoints
# ------------------------------------------------------------------------------

@app.post("/fer/capture")
def fer_capture_endpoint(payload: FERCaptureRequest):
    """
    Captures video from webcam and returns detected dominant emotion.
    
    This endpoint will:
    1. Access the user's webcam
    2. Capture video for the specified duration
    3. Analyze frames for emotion detection
    4. Return the dominant emotion detected
    
    Note: This requires the server to have access to a webcam.
    For production use, consider capturing video on the client side.
    
    Request JSON:
    {
      "duration_seconds": int (default: 5)
    }
    
    Response JSON:
    {
      "emotion": str,
      "duration_seconds": int
    }
    """
    try:
        logger.info(f"Starting FER capture for {payload.duration_seconds} seconds")
        emotion = capture_emotion_from_video(duration_seconds=payload.duration_seconds)
        logger.info(f"FER capture complete. Detected emotion: {emotion}")
        
        return {
            "emotion": emotion,
            "duration_seconds": payload.duration_seconds
        }
    except Exception as e:
        logger.exception("Error in /fer/capture endpoint")
        raise HTTPException(status_code=500, detail=f"FER capture failed: {str(e)}")


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------

@app.post("/therapist")
def therapist_endpoint(payload: TherapistRequest):
    """
    Wraps multimodel_therapist.get_therapist_response to produce a therapist-style response.

    Request JSON:
    {
      "query": str,
      "stress": float,
      "mood": str,
      "fatigue": float,
      "recovery": float,
      "fer_mood": str
    }

    Response JSON:
    {
      "response": str
    }
    """
    try:
        logger.info("Received /therapist request")
        response_text = get_therapist_response(
            query=payload.query,
            stress=payload.stress,
            mood=payload.mood,
            fatigue=payload.fatigue,
            recovery=payload.recovery,
            fer_mood=payload.fer_mood,
        )
        return {"response": response_text}
    except Exception as e:
        logger.exception("Error in /therapist endpoint")
        raise HTTPException(status_code=500, detail=f"Therapist generation failed: {str(e)}")


@app.post("/friend")
def friend_endpoint(payload: FriendRequest):
    """
    Wraps multimodel_friend.get_friend_response to produce a best-friend style response.

    Request JSON:
    {
      "query": str,
      "mode": str,
      "friend_name": str,
      "fer_emotion": str (optional, default: "neutral")
    }

    Response JSON:
    {
      "response": str
    }
    """
    try:
        logger.info("Received /friend request")
        response_text = get_friend_response(
            query=payload.query,
            mode=payload.mode,
            friend_name=payload.friend_name,
            fer_emotion=payload.fer_emotion,
        )
        return {"response": response_text}
    except Exception as e:
        logger.exception("Error in /friend endpoint")
        raise HTTPException(status_code=500, detail=f"Friend generation failed: {str(e)}")


@app.post("/friend/with-fer")
def friend_with_fer_endpoint(payload: FriendWithFERRequest):
    """
    Captures FER emotion first (5 seconds of video) then generates friend response.
    This is a convenience endpoint that combines /fer/capture and /friend.
    
    WORKFLOW:
    1. Capture video from webcam for specified duration
    2. Detect dominant emotion
    3. Use detected emotion + user query to generate friend response
    
    Request JSON:
    {
      "query": str,
      "mode": str,
      "friend_name": str,
      "fer_duration": int (default: 5)
    }
    
    Response JSON:
    {
      "response": str,
      "detected_emotion": str
    }
    """
    try:
        logger.info(f"Received /friend/with-fer request - capturing emotion for {payload.fer_duration}s")
        
        # Step 1: Capture emotion
        detected_emotion = capture_emotion_from_video(duration_seconds=payload.fer_duration)
        logger.info(f"Detected emotion: {detected_emotion}")
        
        # Step 2: Generate friend response with detected emotion
        response_text = get_friend_response(
            query=payload.query,
            mode=payload.mode,
            friend_name=payload.friend_name,
            fer_emotion=detected_emotion,
        )
        
        return {
            "response": response_text,
            "detected_emotion": detected_emotion
        }
    except Exception as e:
        logger.exception("Error in /friend/with-fer endpoint")
        raise HTTPException(status_code=500, detail=f"Friend with FER generation failed: {str(e)}")


# ------------------------------------------------------------------------------
# Run Instructions
# ------------------------------------------------------------------------------
# Save this file as app.py (or ensure the module is named 'app' exposing 'app' variable).
# Then run the server with:
#   uvicorn app:app --reload
#
# The server will expose:
# - GET  /health
# - POST /therapist
# - POST /friend (with optional fer_emotion parameter)
# - POST /fer/capture (standalone FER capture)
# - POST /friend/with-fer (captures FER then responds - recommended for your use case)
#
# RECOMMENDED WORKFLOW:
# Use the /friend/with-fer endpoint which will:
# 1. Show webcam feed for 5 seconds
# 2. Detect emotion from facial expressions
# 3. Generate contextually appropriate friend response
#
# Example request to /friend/with-fer:
# {
#   "query": "Hey, how's it going?",
#   "mode": "caring",
#   "friend_name": "Sunny",
#   "fer_duration": 5
# }