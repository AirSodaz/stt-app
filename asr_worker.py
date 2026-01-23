import sys
import os
import traceback
import queue
import logging
# Ensure the current directory is in sys.path so we can import asr_inference
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from asr_inference import ASRModelManager

# Configure basic logging for the worker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ASRWorker")

class QueueStderr:
    """
    Custom file-like object to capture stderr and send it to a queue.
    Used to forward tqdm progress bars from the worker to the main process.
    """
    def __init__(self, queue):
        self.queue = queue
        
    def write(self, s):
        # Forward everything. Client side filtering can decide what to keep.
        if s:
            self.queue.put(("PROGRESS", s))
            
    def flush(self):
        pass

def worker_main(input_queue, result_queue):
    """
    Main loop for the ASR worker process.
    """
    import signal
    try:
        # Ignore CTRL+C / CTRL+BREAK in worker so it doesn't crash prematurely
        # The parent process will manage our lifecycle via terminate()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    except Exception:
        pass

    logger.info("ASR Worker process started.")
    
    asr_manager = None
    
    try:
        asr_manager = ASRModelManager()
    except Exception as e:
        logger.critical(f"Failed to initialize ASRModelManager in worker: {e}")
        result_queue.put(("INIT_ERROR", str(e)))
        return

    result_queue.put(("INIT_SUCCESS", None))

    while True:
        try:
            # Block until a command is received
            cmd_data = input_queue.get()
            
            if cmd_data is None:
                # Sentinel to exit
                logger.info("Received exit signal. Shutting down worker.")
                break
                
            command, args, kwargs = cmd_data
            
            try:
                if command == "LOAD":
                    # args: (model_name,)
                    model_name = args[0]
                    logger.info(f"Worker processing LOAD: {model_name}")
                    asr_manager.load_model(model_name)
                    result_queue.put(("SUCCESS", model_name))
                    
                elif command == "TRANSCRIBE":
                    # kwargs: {audio_input, language, use_itn, model_name, show_emoji}
                    logger.info(f"Worker processing TRANSCRIBE")
                    result = asr_manager.transcribe(**kwargs)
                    result_queue.put(("SUCCESS", result))

                elif command == "INFERENCE_CHUNK":
                    # kwargs: {audio_chunk, cache, is_final, model_name}
                    # logger.debug(f"Worker processing INFERENCE_CHUNK")
                    result = asr_manager.inference_chunk(**kwargs)
                    result_queue.put(("SUCCESS", result))
                    
                elif command == "DOWNLOAD":
                    # args: (model_name,)
                    model_name = args[0]
                    logger.info(f"Worker processing DOWNLOAD: {model_name}")
                    
                    # Redirect stderr to capture tqdm output
                    old_stderr = sys.stderr
                    sys.stderr = QueueStderr(result_queue)
                    try:
                        path = asr_manager.download_model(model_name)
                        result_queue.put(("SUCCESS", path))
                    except Exception as e:
                        raise e
                    finally:
                        sys.stderr = old_stderr

                elif command == "GET_DOWNLOADED":
                     models = asr_manager.get_downloaded_models()
                     result_queue.put(("SUCCESS", models))

                elif command == "PING":
                    result_queue.put(("PONG", None))
                    
                else:
                    logger.error(f"Unknown command: {command}")
                    result_queue.put(("ERROR", f"Unknown command: {command}"))
                    
            except Exception as e:
                logger.error(f"Error processing command {command}: {e}")
                # Send error back
                result_queue.put(("ERROR", str(e)))
                
        except KeyboardInterrupt:
            logger.info("Worker interrupted.")
            break
        except Exception as e:
            logger.critical(f"Worker main loop error: {e}")
            break
            
    logger.info("ASR Worker process exiting.")

if __name__ == "__main__":
    # This block is only run if the script is executed directly, 
    # but normally it will be imported or launched via multiprocessing.Process target.
    pass
