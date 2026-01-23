import multiprocessing
import queue
import threading
import logging
import os
import sys
# Import the worker function
from asr_worker import worker_main

logger = logging.getLogger("ASRClient")

class ASRClient:
    """
    Client to manage the ASR worker process.
    Handles interruptions by restarting the worker process.
    """
    
    MODEL_TYPES = {
        "SenseVoiceSmall": "offline",
        "Paraformer": "offline",
        "Paraformer-Online": "online",
        "Fun-ASR-Nano": "offline",
    }
    
    MODELS = {
         "SenseVoiceSmall": "iic/SenseVoiceSmall",
         "Paraformer": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
         "Paraformer-Online": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
         "Fun-ASR-Nano": "FunAudioLLM/Fun-ASR-Nano-2512",
    }

    def __init__(self):
        self.lock = threading.RLock()
        self.current_model_name = None
        self.worker_process = None
        self.input_queue = None
        self.result_queue = None
        self.busy = False  # Track if worker is currently processing a command
        self.current_task_type = None # "LOAD", "TRANSCRIBE", "DOWNLOAD"
        self._ensure_worker()

    def _ensure_worker(self):
        """Ensure the worker process is running."""
        with self.lock:
            if self.worker_process is None or not self.worker_process.is_alive():
                logger.info("Starting ASR worker process...")
                self.input_queue = multiprocessing.Queue()
                self.result_queue = multiprocessing.Queue()
                self.worker_process = multiprocessing.Process(
                    target=worker_main, 
                    args=(self.input_queue, self.result_queue)
                )
                self.worker_process.daemon = True # Ensure it dies if parent dies
                self.worker_process.start()
                
                # Wait for init
                try:
                    # Windows spawn can be slow due to heavy imports (PyTorch, etc.)
                    res = self.result_queue.get(timeout=60)
                    if res[0] == "INIT_SUCCESS":
                        logger.info("ASR worker started successfully.")
                    else:
                        logger.error(f"ASR worker init failed: {res}")
                except queue.Empty:
                    logger.error("ASR worker init timed out (60s).")
                
                self.busy = False
                self.current_task_type = None


    def _restart_worker(self):
        """
        Forcefully restart the worker.
        This effectively cancels any running task.
        """
        with self.lock:
            if self.worker_process and self.worker_process.is_alive():
                logger.warning("Terminating ASR worker process...")
                self.worker_process.terminate()
                
                # Notify any waiting threads that the worker is gone
                try:
                    while not self.result_queue.empty():
                         self.result_queue.get_nowait()
                    self.result_queue.put(("ERROR", "Worker interrupted/restarted"))
                except Exception:
                    pass
                
                self.worker_process.join(timeout=1)
                if self.worker_process.is_alive():
                     logger.warning("Force killing ASR worker...")
                     self.worker_process.kill()
                     
            self.worker_process = None
            self.current_model_name = None
            self.busy = False
            self.current_task_type = None
            self._ensure_worker()

    def load_model(self, model_name: str):
        """
        Load a model. 
        Optimization: 
        - If busy with TRANSCRIPTION, restart worker (interrupt).
        - If busy with LOADING, do NOT restart (just queue), because restart cost > load cost.
        """
        with self.lock:
            # If requesting the same model and we are healthy
            if self.current_model_name == model_name:
                if self.worker_process and self.worker_process.is_alive():
                     logger.info(f"Model '{model_name}' already current.")
                     return
            
            # Logic for interruption
            should_restart = False
            
            if self.worker_process is None or not self.worker_process.is_alive():
                should_restart = True
            elif self.busy:
                if self.current_task_type == "TRANSCRIBE" or self.current_task_type == "DOWNLOAD":
                    # Interrupt long-running tasks
                    logger.info(f"Worker busy with {self.current_task_type}. Restarting to interrupt.")
                    should_restart = True
                else:
                    # If busy with LOAD or short task, wait (queueing is faster than restart)
                    logger.info(f"Worker busy with {self.current_task_type}. Queueing load request (faster than restart).")
                    should_restart = False
            else:
                 # Idle
                 should_restart = False

            if should_restart:
                self._restart_worker()
            
            self.busy = True
            self.current_task_type = "LOAD"
            self.current_model_name = model_name
            self.input_queue.put(("LOAD", (model_name,), {}))
            
            # Capture queue to wait on (thread-safety locally)
            res_q = self.result_queue
            
        # Wait for result outside lock
        try:
            status, payload = res_q.get()
        finally:
            with self.lock:
                # Only reset busy if the finished task matches what we started?
                # Simple boolean is rough but okay for sequential queue usage.
                self.busy = not self.input_queue.empty()
                if not self.busy:
                    self.current_task_type = None

        if status == "ERROR":
            raise RuntimeError(f"Load failed: {payload}")
        if status != "SUCCESS":
             raise RuntimeError(f"Unexpected load response: {status}, {payload}")
             
        logger.info(f"Model '{model_name}' loaded successfully.")


    def transcribe(self, audio_input, language="auto", use_itn=False, model_name=None, show_emoji=True):
        """
        Transcribe audio. IF `model_name` is different, it will trigger a restart/load (via helper).
        """
        target_model = model_name if model_name else self.current_model_name
        if not target_model:
             raise ValueError("No model specified.")
             
        # Check if we need to load (and thus interrupt)
        with self.lock:
             if self.current_model_name != target_model:
                  self.load_model(target_model)
                  
             # Now send transcribe command
             self.busy = True
             self.current_task_type = "TRANSCRIBE"
             self.input_queue.put(("TRANSCRIBE", (), {
                 "audio_input": audio_input,
                 "language": language,
                 "use_itn": use_itn,
                 "model_name": target_model, # Worker uses this to verify or just use loaded
                 "show_emoji": show_emoji
             }))
             res_q = self.result_queue
             
        # Wait for result
        try:
            status, payload = res_q.get()
        finally:
            with self.lock:
                self.busy = not self.input_queue.empty()
                if not self.busy:
                    self.current_task_type = None
                
        if status == "ERROR":
            raise RuntimeError(f"Transcription failed: {payload}")
        return payload

    def inference_chunk(self, audio_chunk, cache=None, is_final=False, model_name=None):
        """
        Streaming inference chunk.
        """
        target_model = model_name if model_name else self.current_model_name
        if not target_model:
             raise ValueError("No model specified.")
             
        with self.lock:
             if self.current_model_name != target_model:
                  self.load_model(target_model)
             
             # Streaming is tricky for "busy". 
             self.busy = True 
             self.current_task_type = "TRANSCRIBE" # Treat streaming as transcribe for interruption purposes
             self.input_queue.put(("INFERENCE_CHUNK", (), {
                 "audio_chunk": audio_chunk,
                 "cache": cache,
                 "is_final": is_final,
                 "model_name": target_model
             }))
             res_q = self.result_queue
             
        try:
            status, payload = res_q.get()
        finally:
             with self.lock:
                  self.busy = not self.input_queue.empty()
                  if not self.busy:
                      self.current_task_type = None
        
        if status == "ERROR":
            raise RuntimeError(f"Inference chunk failed: {payload}")
        return payload

    def download_model(self, model_name, progress_callback=None):
        """
        Download model with progress feedback.
        """
        with self.lock:
            self._ensure_worker() # Ensure alive
            
            # Mark as busy with download (interruptible)
            self.busy = True
            self.current_task_type = "DOWNLOAD"
            
            self.input_queue.put(("DOWNLOAD", (model_name,), {}))
            res_q = self.result_queue
            
        # Wait for stream of PROGRESS messages until SUCCESS or ERROR
        try:
            while True:
                status, payload = res_q.get()
                if status == "PROGRESS":
                    if progress_callback:
                        progress_callback(payload)
                elif status == "SUCCESS":
                    return payload
                elif status == "ERROR":
                     raise RuntimeError(f"Download failed: {payload}")
                else:
                     logger.warning(f"Unknown message during download: {status}")
        finally:
            with self.lock:
                self.busy = not self.input_queue.empty()
                if not self.busy:
                    self.current_task_type = None

    def get_downloaded_models(self):
        with self.lock:
            self._ensure_worker()
            self.input_queue.put(("GET_DOWNLOADED", (), {}))
            res_q = self.result_queue
            
        status, payload = res_q.get()
        if status == "ERROR":
             raise RuntimeError(f"Get downloaded failed: {payload}")
        return payload

    def is_model_downloaded(self, model_name):
         # This needs access to filesystem.
         # Since we share the filesystem, we can check locally without asking worker?
         # YES. `ASRModelManager` logic relies on `models` dir.
         
         if model_name not in self.MODELS:
              return False
         model_id = self.MODELS[model_name]
         # Hardcoded path assumption matching `ASRModelManager` defaults
         # Ideally we should get this from config or worker.
         # But `ASRModelManager(model_cache_dir="./models")` is default.
         path = os.path.join("./models", model_id, "config.yaml")
         return os.path.exists(path)

    def get_model_type(self, model_name):
        return self.MODEL_TYPES.get(model_name, "offline")
