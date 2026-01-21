import subprocess
import sys
import os
import time
import signal
import atexit
from logger import setup_logging

# Setup logger for this script
logger = setup_logging("StartDev")

# Global process references for cleanup
_backend_proc = None
_frontend_proc = None

def _kill_process_tree(pid):
    """Kill a process and all its children (Windows & Unix compatible)."""
    try:
        if os.name == 'nt':
            # Windows: use taskkill with /T flag to kill process tree
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(pid)],
                capture_output=True,
                timeout=10
            )
        else:
            # Unix: send SIGTERM to process group
            import signal
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception as e:
        logger.warning(f"[Start Script] Failed to kill process tree {pid}: {e}")

def _cleanup_on_exit():
    """Cleanup function registered with atexit to ensure processes are terminated."""
    global _backend_proc, _frontend_proc
    
    if _backend_proc is not None and _backend_proc.poll() is None:
        logger.info("[Start Script] atexit: Shutting down backend...")
        try:
            if os.name == 'nt':
                _backend_proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                _backend_proc.terminate()
            _backend_proc.wait(timeout=5)
        except Exception as e:
            logger.warning(f"[Start Script] atexit: Failed graceful shutdown, killing tree: {e}")
            _kill_process_tree(_backend_proc.pid)
    
    if _frontend_proc is not None and _frontend_proc.poll() is None:
        logger.info("[Start Script] atexit: Terminating frontend process tree...")
        # Frontend (npm -> vite -> tauri) needs tree kill to clean up all children
        _kill_process_tree(_frontend_proc.pid)

# Register cleanup function
atexit.register(_cleanup_on_exit)

def shutdown_backend_gracefully(backend_proc):
    if backend_proc.poll() is None:
        logger.info("[Start Script] Sending CTRL_BREAK_EVENT to backend...")
        try:
            # Use CTRL_BREAK_EVENT instead of CTRL_C_EVENT on Windows.
            # When a process is created with CREATE_NEW_PROCESS_GROUP,
            # CTRL_C_EVENT won't reach it, but CTRL_BREAK_EVENT will.
            backend_proc.send_signal(signal.CTRL_BREAK_EVENT)
        except Exception as e:
            logger.error(f"[Start Script] Failed to send signal: {e}")
        
        # Wait for backend to exit gracefully
        t_end = time.time() + 15
        while time.time() < t_end:
            if backend_proc.poll() is not None:
                logger.info("[Start Script] Backend exited gracefully.")
                return
            time.sleep(0.1)
        
        logger.warning("[Start Script] Backend did not exit in time. Terminating...")
        backend_proc.terminate()

def main():
    global _backend_proc, _frontend_proc
    
    # Get the directory where this script is located
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ui_dir = os.path.join(root_dir, "ui")

    logger.info("[Start Script] Launching Backend Server...")
    # Start backend in a new process group so we can send CTRL_C_EVENT to it specifically
    # without killing the script itself immediately, and handle signals manually.
    kwargs = {}
    if os.name == 'nt':
        kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP

    backend = subprocess.Popen([sys.executable, "server.py"], cwd=root_dir, **kwargs)
    _backend_proc = backend  # Store reference for atexit cleanup

    # Give the backend a moment to initialize
    time.sleep(2)

    logger.info("[Start Script] Launching Tauri Frontend...")
    # Determine the npm command based on the OS
    npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"
    
    # Start the frontend
    frontend = subprocess.Popen([npm_cmd, "run", "tauri", "dev"], cwd=ui_dir)
    _frontend_proc = frontend  # Store reference for atexit cleanup

    try:
        # Keep the script running to monitor child processes
        while True:
            time.sleep(1)
            # Check if backend has exited
            if backend.poll() is not None:
                logger.error("[Start Script] Backend server exited unexpectedly.")
                break
            # Check if frontend has exited
            if frontend.poll() is not None:
                logger.info("[Start Script] Frontend exited.")
                break
    except KeyboardInterrupt:
        logger.info("\n[Start Script] Stopping processes (Ctrl+C received)...")
        # On KeyboardInterrupt, we also want to try graceful shutdown
        shutdown_backend_gracefully(backend)
    finally:
        # If we broke out of the loop (e.g. frontend closed), backend might still be running
        if backend.poll() is None:
            logger.info("[Start Script] Shutting down backend...")
            shutdown_backend_gracefully(backend)
            
        if frontend.poll() is None:
            logger.info("[Start Script] Terminating frontend process tree...")
            _kill_process_tree(frontend.pid)
            
        logger.info("[Start Script] Shutdown complete.")

if __name__ == "__main__":
    main()
