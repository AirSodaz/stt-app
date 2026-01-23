from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import shutil
import os
import uuid
import logging
import stat
import json
import asyncio
import threading
from typing import Dict, Any, List, Optional, Generator
from asr_client import ASRClient

# Configure logging
from logger import setup_logging

# Configure logging
logger = setup_logging(__name__)


# Global ASR Client
asr_client: Optional[ASRClient] = None

TEMP_DIR = "temp_uploads"

# Download progress tracking
download_progress: Dict[str, Dict[str, Any]] = {}
# Format: {"model_name": {"status": "downloading"|"complete"|"error", "progress": 0-100, "message": "..."}}

# Model loading state tracking
model_loading_state: Dict[str, Any] = {
    "is_loading": False,
    "loading_model": None,
    "loaded_model": None,
    "error": None
}

def update_download_progress(model_name: str, status: str, progress: int, message: str = "") -> None:
    """Update download progress for a model."""
    download_progress[model_name] = {
        "status": status,
        "progress": progress,
        "message": message
    }

def save_uploaded_file(file_obj, dest_path: str) -> None:
    """Helper to save uploaded file to disk in a thread-safe way."""
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file_obj, buffer)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_client
    # Startup: Ensure temp dir exists (and clean it first)
    if os.path.exists(TEMP_DIR):
        try:
            # Helper to handle read-only files (common on Windows)
            def remove_readonly(func, path, excinfo):
                os.chmod(path, stat.S_IWRITE)
                func(path)
                
            shutil.rmtree(TEMP_DIR, onerror=remove_readonly)
            logger.info(f"Startup: Cleaned up existing temp directory: {TEMP_DIR}")
        except Exception as e:
            logger.warning(f"Startup: Failed to clean up existing temp directory {TEMP_DIR}: {e}")

    os.makedirs(TEMP_DIR, exist_ok=True)
    logger.info(f"Created temp directory: {TEMP_DIR}")
    
    # Initialize ASR Client
    # Only initialize if we are NOT in the worker process (which shouldn't happen here anyway as we are in server.py)
    # But crucially, this code only runs when FastAPI starts, effectively guarding it.
    try:
        logger.info("Startup: Initializing ASR Client...")
        asr_client = ASRClient()
    except Exception as e:
        logger.critical(f"Failed to initialize ASRClient: {e}")
        asr_client = None
    
    logger.info("Startup: Server ready to accept requests.")
            
    yield
    
    # Shutdown logic
    logger.info("Shutdown: Stopping ASR Client...")
    if asr_client:
        # We should add a stop/shutdown method to ASRClient to be clean
        # For now, just letting it be garbage collected or killed by OS might be okay,
        # but explicit termination is better.
        try:
            if asr_client.worker_process and asr_client.worker_process.is_alive():
                asr_client.worker_process.terminate()
                asr_client.worker_process.join(timeout=2)
        except Exception as e:
            logger.error(f"Error shutting down ASRClient: {e}")

    # Shutdown: Clean up temp dir
    if os.path.exists(TEMP_DIR):
        try:
            # Helper to handle read-only files (common on Windows)
            def remove_readonly(func, path, excinfo):
                os.chmod(path, stat.S_IWRITE)
                func(path)
                
            shutil.rmtree(TEMP_DIR, onerror=remove_readonly)
            logger.info(f"Cleaned up temp directory: {TEMP_DIR}")
        except Exception as e:
            logger.error(f"Failed to clean up temp directory {TEMP_DIR}: {e}")

app = FastAPI(
    title="SenseVoice API",
    description="API for SenseVoice speech-to-text inference",
    version="1.0.0",
    lifespan=lifespan
)

# Allow CORS for Tauri app (usually runs on localhost:1420)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/models", response_model=List[Dict[str, Any]])
async def get_models() -> List[Dict[str, Any]]:
    """
    Get list of available ASR models with their download status.
    """
    if asr_client is None:
        raise HTTPException(status_code=500, detail="ASR Client not initialized.")
    
    models_status = []
    for name in asr_client.MODELS.keys():
        models_status.append({
            "name": name,
            "type": asr_client.get_model_type(name),
            "downloaded": asr_client.is_model_downloaded(name)
        })
    return models_status


@app.get("/models/downloaded", response_model=List[str])
async def get_downloaded_models() -> List[str]:
    """
    Get list of downloaded model names only.
    """
    if asr_client is None:
         # Attempt lazy init if it failed during startup? No, that's unsafe.
        raise HTTPException(status_code=500, detail="ASR Client not initialized.")
    
    return asr_client.get_downloaded_models()


@app.get("/models/status")
async def get_model_status() -> Dict[str, Any]:
    """
    Get the current model loading status.
    Returns:
        - loaded_model: Name of the currently loaded model, or null
        - is_loading: Whether a model is currently being loaded
        - loading_model: Name of the model being loaded (if is_loading is true)
        - error: Error message if loading failed
    """
    return {
        "loaded_model": model_loading_state["loaded_model"],
        "is_loading": model_loading_state["is_loading"],
        "loading_model": model_loading_state["loading_model"],
        "error": model_loading_state["error"]
    }


@app.post("/load_model")
async def load_model_endpoint(model_name: str = Form(...)) -> Dict[str, str]:
    """
    Trigger background loading of a model.
    Returns immediately while model loads in background.
    Use GET /models/status to check loading progress.
    """
    if asr_client is None:
        raise HTTPException(status_code=500, detail="ASR Client not initialized.")
    
    if model_name not in asr_client.MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    
    # Check if model is downloaded
    if not asr_client.is_model_downloaded(model_name):
        raise HTTPException(status_code=400, detail=f"Model not downloaded: {model_name}")
    
    # Check if model is already loaded (and worker is alive)
    # asr_client.load_model handles logic, but this quick check avoids thread spawn if possible.
    # However, for interruption support, client.load_model is the source of truth.
    # If we are effectively "switching", client.load_model does the work.
    
    # Start loading in background thread
    def load_model_background():
        try:
            model_loading_state["is_loading"] = True
            model_loading_state["loading_model"] = model_name
            model_loading_state["error"] = None
            
            logger.info(f"Background loading model: {model_name}")
            # This call blocks until loaded (or interrupted/errored)
            # It triggers the worker restart if needed.
            asr_client.load_model(model_name)
            
            # If successful (didn't raise exception)
            model_loading_state["loaded_model"] = model_name
            model_loading_state["is_loading"] = False
            model_loading_state["loading_model"] = None
            logger.info(f"Background loading complete: {model_name}")
            
        except RuntimeError as e:
            # Check if it was an interruption or actual error?
            # ASRClient raises RuntimeError("Load failed: ...") or "Worker restarted"
            # If interrupted by another task, we might not want to show "Error" but just "Cancelled"?
            # But here we are the thread that initiated it. If another thread interrupted us, 
            # we failed to complete OUR task.
            logger.error(f"Background loading failed for {model_name}: {e}")
            model_loading_state["is_loading"] = False
            model_loading_state["loading_model"] = None
            model_loading_state["error"] = str(e)
        except Exception as e:
            logger.error(f"Background loading failed for {model_name}: {e}")
            model_loading_state["is_loading"] = False
            model_loading_state["loading_model"] = None
            model_loading_state["error"] = str(e)
    
    thread = threading.Thread(target=load_model_background)
    thread.daemon = True
    thread.start()
    
    return {"status": "loading", "message": f"Model loading started: {model_name}"}


def parse_progress_text(model_name: str, text: str):
    """Parse tqdm output and update progress"""
    import re
    # Match: "Downloading [model.pt]: 45%|███| 402M/893M [00:15<00:18, 25.7MB/s]"
    match = re.search(r'Downloading \[model\.pt\]:\s*(\d+)%.*?(\d+\.?\d*[kMG]?)/(\d+\.?\d*[kMG]?)', text)
    if match:
        percent = int(match.group(1))
        downloaded = match.group(2)
        total = match.group(3)
        update_download_progress(
            model_name,
            "downloading",
            percent,
            f"Downloading model.pt... {percent}% ({downloaded}/{total})"
        )

def run_download_with_progress(model_name: str) -> None:
    """Run model download in background, capturing progress via callback."""
    if asr_client is None:
         return

    try:
        update_download_progress(model_name, "downloading", 0, "Initializing download...")
        
        def progress_callback(text):
             parse_progress_text(model_name, text)

        # Run the download
        asr_client.download_model(model_name, progress_callback=progress_callback)
        
        # Complete
        update_download_progress(model_name, "complete", 100, "Download complete!")
        logger.info(f"Model '{model_name}' downloaded successfully.")
        
    except Exception as e:
        logger.error(f"Download failed for {model_name}: {e}")
        update_download_progress(model_name, "error", 0, f"Download failed: {str(e)}")


async def generate_download_progress(model_name: str) -> Generator[str, None, None]:
    """Generate SSE events for download progress."""
    last_status = None
    last_progress = -1
    
    while True:
        if model_name in download_progress:
            current = download_progress[model_name]
            
            # Only send if changed
            if current["status"] != last_status or current["progress"] != last_progress:
                last_status = current["status"]
                last_progress = current["progress"]
                
                event_data = json.dumps(current)
                yield f"data: {event_data}\n\n"
                
                # Stop streaming if complete or error
                if current["status"] in ["complete", "error"]:
                    # Clean up
                    del download_progress[model_name]
                    break
        
        await asyncio.sleep(0.2)


@app.get("/download_model/{model_name}")
async def download_model_with_progress(model_name: str):
    """
    Start model download and stream progress via SSE.
    """
    if asr_client is None:
        raise HTTPException(status_code=500, detail="ASR Client not initialized.")
    
    if model_name not in asr_client.MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
    
    # Check if already downloading
    if model_name in download_progress and download_progress[model_name]["status"] == "downloading":
        pass  # Continue streaming existing download
    else:
        # Start download in background thread
        thread = threading.Thread(target=run_download_with_progress, args=(model_name,))
        thread.daemon = True
        thread.start()
    
    return StreamingResponse(
        generate_download_progress(model_name),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/download_model")
async def download_model(model_name: str = Form(...)) -> Dict[str, str]:
    """
    Trigger download of a specific model (legacy endpoint, sync).
    """
    if asr_client is None:
        raise HTTPException(status_code=500, detail="ASR Client not initialized.")
    
    if model_name not in asr_client.MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    try:
        asr_client.download_model(model_name)
        return {"status": "success", "message": f"Model '{model_name}' downloaded successfully."}
    except Exception as e:
        logger.error(f"Download failed for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.post("/transcribe", response_model=Dict[str, str])
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    use_itn: bool = Form(False),
    model: str = Form(None),
    show_emoji: bool = Form(True)
) -> Dict[str, str]:
    """
    Transcribe an uploaded audio file.

    Args:
        file (UploadFile): The audio file to transcribe.
        language (str): The target language (default: "auto").
        use_itn (bool): WHETHER to use Inverse Text Normalization (default: False).
        model (str): Optional model name to use for this transcription.
    
    Returns:
        Dict[str, str]: A dictionary containing the transcribed text key "text".
    
    Raises:
        HTTPException: If model is not loaded or transcription fails.
    """
    if asr_client is None:
        logger.error("ASR Client is not initialized.")
        raise HTTPException(status_code=500, detail="Speech-to-text model manager not initialized.")

    # Save uploaded file
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ""
    temp_filename = f"{uuid.uuid4()}{file_extension}"
    temp_file_path = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        await asyncio.to_thread(save_uploaded_file, file.file, temp_file_path)
        
        # DEBUG: Log file details
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Received file: {file.filename}, saved to: {temp_file_path}, size: {file_size} bytes")
        
        if file_size == 0:
            logger.warning("Received empty audio file!")
            return {"text": ""}
            
        logger.info(f"Transcribing {temp_file_path} with language={language}, use_itn={use_itn}, model={model}, show_emoji={show_emoji}")
        
        # Run inference in a separate thread to avoid blocking the event loop
        # asr_client.transcribe handles thread-safety and interruption internally.
        text = await asyncio.to_thread(
            asr_client.transcribe,
            temp_file_path, 
            language=language, 
            use_itn=use_itn, 
            model_name=model,
            show_emoji=show_emoji
        )
        
        # DEBUG: Log the result
        logger.info(f"Transcription result: '{text[:100]}...'" if len(text) > 100 else f"Transcription result: '{text}'")
        
        return {"text": text}
        
    except Exception as e:
        logger.error(f"Error during transcription request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Removed temp file: {temp_file_path}")
            except OSError as e:
                logger.warning(f"Failed to remove temp file {temp_file_path}: {e}")

@app.post("/transcribe_stream")
async def transcribe_audio_stream(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    use_itn: bool = Form(False),
    model: str = Form(None),
    show_emoji: bool = Form(True)
):
    """
    Transcribe an uploaded audio file with SSE progress updates.
    This endpoint sends heartbeat messages during processing to prevent timeout.
    """
    if asr_client is None:
        logger.error("ASR Client is not initialized.")
        raise HTTPException(status_code=500, detail="Speech-to-text model manager not initialized.")

    # Save uploaded file
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ""
    temp_filename = f"{uuid.uuid4()}{file_extension}"
    temp_file_path = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        await asyncio.to_thread(save_uploaded_file, file.file, temp_file_path)
        
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"[SSE] Received file: {file.filename}, size: {file_size} bytes")
        
        if file_size == 0:
            logger.warning("[SSE] Received empty audio file!")
            async def empty_response():
                yield f"data: {json.dumps({'status': 'complete', 'text': ''})}\n\n"
            return StreamingResponse(empty_response(), media_type="text/event-stream")
            
    except Exception as e:
        logger.error(f"[SSE] Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # Create a result container for the background thread
    result_container = {"text": None, "error": None, "done": False}
    
    def run_transcription():
        """Run transcription in background thread."""
        try:
            logger.info(f"[SSE] Starting transcription with model={model}")
            text = asr_client.transcribe(
                temp_file_path,
                language=language,
                use_itn=use_itn,
                model_name=model,
                show_emoji=show_emoji
            )
            result_container["text"] = text
            logger.info(f"[SSE] Transcription complete: '{text[:50]}...'" if len(text) > 50 else f"[SSE] Transcription complete: '{text}'")
        except Exception as e:
            logger.error(f"[SSE] Transcription error: {e}")
            result_container["error"] = str(e)
        finally:
            result_container["done"] = True
    
    async def generate_sse_events():
        """Generate SSE events with heartbeat during processing."""
        # Start transcription in background
        thread = threading.Thread(target=run_transcription)
        thread.start()
        
        heartbeat_count = 0
        
        # Send processing status and heartbeats until done
        while not result_container["done"]:
            heartbeat_count += 1
            yield f"data: {json.dumps({'status': 'processing', 'heartbeat': heartbeat_count})}\n\n"
            await asyncio.sleep(1)  # Send heartbeat every second
        
        # Send final result
        if result_container["error"]:
            yield f"data: {json.dumps({'status': 'error', 'message': result_container['error']})}\n\n"
        else:
            text = result_container["text"] or ""
            yield f"data: {json.dumps({'status': 'complete', 'text': text})}\n\n"
        
        # Cleanup temp file
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception as e:
            logger.warning(f"[SSE] Failed to cleanup temp file: {e}")
    
    return StreamingResponse(
        generate_sse_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket, model: str = None):
    """
    WebSocket endpoint for real-time transcription.
    Processes audio in segments for near real-time output.
    
    Query Parameters:
        model (str): Optional model name for streaming. Defaults to first available online model.
    """
    logger.info(f"WS connection attempt with model={model}...")
    await websocket.accept()
    logger.info("WS accepted")
    
    import numpy as np
    
    # Determine which streaming model to use
    streaming_model = None
    # Determine which streaming model to use
    streaming_model = None
    if model and model in asr_client.MODELS:
        # Validate it's an online model
        if asr_client.get_model_type(model) == "online":
            streaming_model = model
        else:
            logger.warning(f"Requested model '{model}' is not a streaming model, falling back to default")
    
    # Fallback: find first available online model
    if not streaming_model:
        for name in asr_client.MODELS.keys():
            if asr_client.get_model_type(name) == "online":
                streaming_model = name
                break
    
    if not streaming_model:
        logger.error("No online streaming model available")
        await websocket.send_json({"error": "No streaming model available"})
        await websocket.close()
        return
    
    logger.info(f"[WS] Using streaming model: {streaming_model}")
    
    # Accumulate audio chunks
    audio_buffer = bytearray()
    SAMPLE_RATE = 16000
    # Streaming chunk size is typically 600ms (10 * 60ms)
    # We set a slightly larger buffer to ensure we have enough data
    SEGMENT_DURATION = 1.2 
    SEGMENT_BYTES = int(SAMPLE_RATE * SEGMENT_DURATION * 2)  # *2 for int16 bytes
    
    # helper to process result
    def process_result(res):
        if isinstance(res, list) and len(res) > 0:
            return res[0].get("text", "")
        return ""

    # Streaming state
    cache = {} 
    all_text = []

    try:
        while True:
            try:
                data = await websocket.receive_bytes()
            except RuntimeError:
                break
                
            if len(data) == 0:
                # EOF signal
                logger.info(f"[WS] EOF received, processing remaining {len(audio_buffer)} bytes...")
                if len(audio_buffer) > 0:
                   audio_data = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                   audio_data = audio_data.astype(np.float32) / 32768.0
                   
                   try:
                       # Final chunk
                       res = await asyncio.to_thread(
                           asr_client.inference_chunk,
                           audio_data, 
                           cache=cache,
                           is_final=True,
                           model_name=streaming_model
                       )
                       text = process_result(res)
                       if text:
                           all_text.append(text)
                   except Exception as e:
                       logger.error(f"[WS] Final chunk error: {e}")
                
                # Send final result
                final_text = "".join(all_text)
                logger.info(f"[WS] Final result: '{final_text}'")
                await websocket.send_json({"text": final_text, "is_final": True})
                break
            
            # Accumulate
            audio_buffer.extend(data)
            
            # Process chunks
            while len(audio_buffer) >= SEGMENT_BYTES:
                # Extract chunk
                num_samples = SEGMENT_BYTES // 2
                
                # Zero-copy view of the data
                audio_view = np.frombuffer(audio_buffer, count=num_samples, dtype=np.int16)

                # Convert to float (creates new array and copy)
                audio_data = audio_view.astype(np.float32) / 32768.0

                # Release the view so we can resize the buffer
                del audio_view

                # In-place remove processed bytes
                del audio_buffer[:SEGMENT_BYTES]
                
                try:
                    res = await asyncio.to_thread(
                        asr_client.inference_chunk,
                        audio_data,
                        cache=cache,
                        is_final=False,
                        model_name=streaming_model
                    )
                    text = process_result(res)
                    if text:
                        all_text.append(text)
                        # Send partial update
                        current_text = "".join(all_text)
                        # logger.info(f"[WS] Partial: {text}") 
                        await websocket.send_json({"text": current_text, "is_final": False})
                    else:
                         # Keep alive
                         await websocket.send_json({"text": "".join(all_text), "is_final": False})
                
                except Exception as e:
                    logger.error(f"[WS] Streaming error: {e}")
                    # Don't break, try to continue

                    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        pass

if __name__ == "__main__":
    import uvicorn
    # Use standard logging config for uvicorn as well if needed, 
    # but basicConfig above handles the app logs.
    uvicorn.run(app, host="127.0.0.1", port=8000)
