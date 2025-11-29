import cv2
import logging
from deepface import DeepFace
from collections import Counter
from typing import Optional

logger = logging.getLogger("fer_module")


def capture_emotion_from_video(duration_seconds: int = 20) -> str:
    """
    Captures video from webcam for specified duration and returns dominant emotion.
    
    Args:
        duration_seconds: Duration to capture video (default 5 seconds)
        
    Returns:
        Dominant emotion detected across the video frames
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Could not open webcam")
        raise RuntimeError("Failed to access webcam")
    
    # Get FPS to calculate total frames needed
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = fps * duration_seconds
    frame_count = 0
    emotions_detected = []
    
    logger.info(f"Starting emotion capture for {duration_seconds} seconds...")
    
    try:
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            # Analyze every 10th frame to improve performance
            if frame_count % 10 == 0:
                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=['emotion'],
                        enforce_detection=False
                    )
                    
                    emotion = result[0]['dominant_emotion']
                    emotions_detected.append(emotion)
                    
                    # Display on frame
                    cv2.putText(
                        frame,
                        f"Emotion: {emotion}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    # Show countdown
                    remaining = duration_seconds - (frame_count / fps)
                    cv2.putText(
                        frame,
                        f"Time: {remaining:.1f}s",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2
                    )
                    
                except Exception as e:
                    logger.debug(f"Frame analysis error: {e}")
                    pass
            
            cv2.imshow("FER Capture - Press 'q' to skip", frame)
            
            # Allow early exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User stopped capture early")
                break
            
            frame_count += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Determine dominant emotion from all detected emotions
    if emotions_detected:
        emotion_counts = Counter(emotions_detected)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        logger.info(f"Detected emotions: {dict(emotion_counts)}")
        logger.info(f"Dominant emotion: {dominant_emotion}")
        return dominant_emotion
    else:
        logger.warning("No emotions detected, returning neutral")
        return "neutral"


def analyze_single_frame(frame) -> Optional[str]:
    """
    Analyzes a single frame for emotion detection.
    
    Args:
        frame: OpenCV frame/image
        
    Returns:
        Detected emotion or None if detection fails
    """
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )
        return result[0]['dominant_emotion']
    except Exception as e:
        logger.error(f"Error analyzing frame: {e}")
        return None


# For testing the module independently
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting FER capture test...")
    emotion = capture_emotion_from_video(duration_seconds=5)
    print(f"\n✅ Final detected emotion: {emotion}")