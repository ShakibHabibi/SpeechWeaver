import whisper
import sounddevice as sd
import numpy as np
from pynput import keyboard
import threading
import queue
import time
import torch
import sys
import gc
import psutil
import os
import subprocess
import pyperclip
from PIL import Image, ImageDraw
import pystray
from PIL import ImageColor
import signal

# Available Whisper models for SpeechWeaver
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

# Global variables for keyboard handling
pressed_keys = set()
stt = None
tray = None

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def get_swap_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().vms / 1024 / 1024  # Convert to MB

def optimize_memory(force=False):
    # Only optimize if memory usage is high or if forced
    current_memory = get_memory_usage()
    if not force and current_memory < 1000:  # Only optimize if memory usage is above 1GB
        return current_memory, get_swap_usage()
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get current memory stats
    ram_usage = get_memory_usage()
    swap_usage = get_swap_usage()
    
    if force:  # Only print stats if forced
        print(f"Memory Usage: {ram_usage:.2f} MB")
        print(f"Swap Usage: {swap_usage:.2f} MB")
    
    return ram_usage, swap_usage

def check_clipboard_dependencies():
    try:
        # Check if clipboard is available without modifying it
        import subprocess
        # Try to run xclip or xsel to check if they're installed
        try:
            subprocess.run(['xclip', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            try:
                subprocess.run(['xsel', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except FileNotFoundError:
                print("\nClipboard functionality is not available.")
                print("To fix this, please install one of the following packages:")
                print("1. xclip: sudo apt-get install xclip")
                print("2. xsel: sudo apt-get install xsel")
                print("\nAfter installing one of these packages, restart the application.")
                return False
    except Exception:
        print("\nClipboard functionality is not available.")
        print("To fix this, please install one of the following packages:")
        print("1. xclip: sudo apt-get install xclip")
        print("2. xsel: sudo apt-get install xsel")
        print("\nAfter installing one of these packages, restart the application.")
        return False

class SpeechWeaver:
    def __init__(self):
        print("Initializing SpeechWeaver...")
        # Check if CUDA is available and use it
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize all attributes first
        self.model = None
        self.current_model = "base"
        self.recording = False
        self.audio_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32
        self.chunk_size = 4096
        self.max_segment_length = 5
        self.max_samples_per_segment = int(self.max_segment_length * self.sample_rate)
        self.audio_buffer = np.array([], dtype=self.dtype)
        self.processing_thread = None
        self.processing_lock = threading.Lock()
        self.chunk_id = 0
        self.last_transcription = ""
        self.last_error = ""
        self.error_lock = threading.Lock()
        
        # Load the initial model after all attributes are initialized
        print("Loading Whisper model...")
        if not self.load_model(self.current_model):
            print("Failed to load initial model. Exiting...")
            sys.exit(1)
            
        print("SpeechWeaver initialized successfully!")
    
    def load_model(self, model_name):
        """Load a Whisper model by name"""
        try:
            print(f"Loading {model_name} model...")
            # Clear existing model from memory
            if self.model is not None:
                del self.model
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            # Load new model
            self.model = whisper.load_model(model_name).to(self.device)
            self.current_model = model_name
            print(f"{model_name} model loaded successfully!")
            return True
        except Exception as e:
            error_msg = f"Error loading model {model_name}: {str(e)}"
            self.log_error(error_msg)
            return False

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        if self.recording:
            try:
                self.audio_queue.put(indata.copy())
            except Exception as e:
                print(f"Error in audio callback: {e}")

    def log_error(self, error_message):
        """Log an error message with timestamp"""
        with self.error_lock:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            self.last_error = f"[{timestamp}] {error_message}"
            print(self.last_error)  # Also print to console

    def process_chunk(self, chunk):
        try:
            # Ensure audio segment is the right shape and type
            if len(chunk.shape) > 1:
                chunk = chunk.squeeze()
            
            # Normalize audio
            max_val = np.max(np.abs(chunk))
            if max_val > 0:
                chunk = chunk / max_val
            
            # Use GPU for transcription with memory optimization
            result = self.model.transcribe(
                chunk,
                fp16=self.device == "cuda",
                language="en",
                task="transcribe"
            )
            
            return result["text"].strip()
        except Exception as e:
            error_msg = f"Error processing chunk: {str(e)}"
            self.log_error(error_msg)
            return ""

    def process_chunks_worker(self):
        while self.recording or not self.processing_queue.empty():
            try:
                # Get chunk from processing queue with timeout
                chunk, chunk_id = self.processing_queue.get(timeout=1.0)
                
                # Process the chunk
                text = self.process_chunk(chunk)
                if text:
                    self.result_queue.put((chunk_id, text))
                
                self.processing_queue.task_done()
                
                # Optimize memory periodically
                if chunk_id % 3 == 0:
                    optimize_memory()
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.log_error(f"Error in processing worker: {str(e)}")

    def record_audio(self):
        print("Starting audio recording...")
        self.chunk_id = 0  # Reset chunk_id at start of recording
        
        try:
            with sd.InputStream(samplerate=self.sample_rate,
                              channels=self.channels,
                              dtype=self.dtype,
                              blocksize=self.chunk_size,
                              callback=self.audio_callback):
                while self.recording:
                    try:
                        # Get audio data from queue
                        audio_data = self.audio_queue.get(timeout=0.1)
                        
                        # Add to buffer
                        self.audio_buffer = np.append(self.audio_buffer, audio_data)
                        
                        # Check if we have enough data for a segment
                        while len(self.audio_buffer) >= self.max_samples_per_segment:
                            # Extract segment
                            segment = self.audio_buffer[:self.max_samples_per_segment]
                            self.audio_buffer = self.audio_buffer[self.max_samples_per_segment:]
                            
                            # Add to processing queue
                            self.processing_queue.put((segment, self.chunk_id))
                            self.chunk_id += 1
                            
                    except queue.Empty:
                        continue
                        
        except Exception as e:
            self.log_error(f"Error in audio recording: {str(e)}")

    def process_audio(self):
        print("Processing remaining audio...")
        try:
            # Process any remaining audio in buffer
            if len(self.audio_buffer) > 0:
                self.processing_queue.put((self.audio_buffer, self.chunk_id))
                self.chunk_id += 1
            
            # Wait for all processing to complete
            self.processing_queue.join()
            
            # Collect results in order
            results = []
            while not self.result_queue.empty():
                chunk_id, text = self.result_queue.get()
                results.append((chunk_id, text))
            
            # Sort by chunk_id and combine text
            results.sort(key=lambda x: x[0])
            full_text = " ".join(text for _, text in results)
            
            optimize_memory(force=True)
            return full_text.strip()
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return ""

    def cancel_process(self):
        """Cancel the current recording and processing"""
        self.recording = False
        
        # Clear all queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear buffer
        self.audio_buffer = np.array([], dtype=self.dtype)
        self.chunk_id = 0
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # Optimize memory
        optimize_memory(force=True)

    def type_text(self, text):
        print(f"Transcribed text: '{text}'")
        try:
            # Store the transcription
            self.last_transcription = text
            
            # Create a keyboard controller
            keyboard_controller = keyboard.Controller()
            # Type the text
            keyboard_controller.type(text)
            print("Text typed successfully")
        except Exception as e:
            print(f"Error typing text: {e}")

class TrayIcon:
    def __init__(self, stt):
        self.stt = stt
        self.icon_size = 64
        self.icon = None
        self.clipboard_available = check_clipboard_dependencies()
        self.setup_icon()
        print("SpeechWeaver system tray icon initialized")
        
    def create_icon(self, color):
        image = Image.new('RGBA', (self.icon_size, self.icon_size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        margin = 4
        draw.ellipse([margin, margin, self.icon_size - margin, self.icon_size - margin], 
                    fill=color)
        return image
    
    def setup_icon(self):
        self.idle_icon = self.create_icon(ImageColor.getrgb('gray'))
        self.recording_icon = self.create_icon(ImageColor.getrgb('green'))
        self.processing_icon = self.create_icon(ImageColor.getrgb('blue'))
        
        # Create model selection submenu
        model_menu_items = []
        for model_name in AVAILABLE_MODELS:
            def make_callback(model):
                def callback():
                    self.change_model(model)
                return callback
            
            model_menu_items.append(
                pystray.MenuItem(
                    model_name,
                    make_callback(model_name),
                    checked=lambda m=model_name: self.stt.current_model == m
                )
            )
        
        # Create main menu items
        menu_items = [
            pystray.MenuItem("Model", pystray.Menu(*model_menu_items)),
            pystray.MenuItem("Start Recording (Ctrl+1)", self.start_recording),
            pystray.MenuItem("Stop Recording (Ctrl+2)", self.stop_recording),
            pystray.MenuItem("Cancel (Ctrl+3)", self.cancel_process),
            pystray.MenuItem("Copy Last Transcription", self.copy_last_transcription),
            pystray.MenuItem("Copy Last Error", self.copy_last_error),
            pystray.MenuItem("Restart Application", self.restart_application),
        ]
        
        if not self.clipboard_available:
            menu_items.append(pystray.MenuItem("⚠️ Clipboard not available", lambda: None))
        
        menu_items.append(pystray.MenuItem("Exit", self.exit_app))
        menu = pystray.Menu(*menu_items)
        
        self.icon = pystray.Icon(
            "speechweaver",
            self.idle_icon,
            f"SpeechWeaver ({self.stt.current_model})",
            menu
        )
        
        self.icon_thread = threading.Thread(target=self.icon.run)
        self.icon_thread.daemon = True
        self.icon_thread.start()
    
    def change_model(self, model_name):
        """Change the current Whisper model"""
        if self.stt.load_model(model_name):
            self.icon.title = f"SpeechWeaver ({model_name})"
            self.icon.update_menu()
    
    def update_icon(self, state):
        if state == "idle":
            self.icon.icon = self.idle_icon
            self.icon.title = f"SpeechWeaver ({self.stt.current_model})"
        elif state == "recording":
            self.icon.icon = self.recording_icon
            self.icon.title = "SpeechWeaver (Recording)"
        elif state == "processing":
            self.icon.icon = self.processing_icon
            self.icon.title = "SpeechWeaver (Processing)"
    
    def start_recording(self):
        if not self.stt.recording:
            self.stt.recording = True
            self.update_icon("recording")
            self.stt.audio_buffer = np.array([], dtype=self.stt.dtype)
            self.stt.processing_queue = queue.Queue()
            self.stt.result_queue = queue.Queue()
            self.stt.chunk_id = 0  # Reset chunk_id when starting new recording
            recording_thread = threading.Thread(target=self.stt.record_audio)
            processing_thread = threading.Thread(target=self.stt.process_chunks_worker)
            recording_thread.start()
            processing_thread.start()
    
    def stop_recording(self):
        if self.stt.recording:
            self.stt.recording = False
            self.update_icon("processing")
            try:
                text = self.stt.process_audio()
                if text:
                    self.stt.type_text(text)
            except Exception as e:
                print(f"Error processing audio: {e}")
            self.update_icon("idle")
    
    def copy_last_transcription(self):
        """Copy the last transcription to clipboard"""
        if self.clipboard_available:
            try:
                pyperclip.copy(self.stt.last_transcription)
                print("Last transcription copied to clipboard")
            except Exception as e:
                print(f"Error copying to clipboard: {e}")
        else:
            print("Clipboard functionality is not available")
    
    def copy_last_error(self):
        """Copy the last error message to clipboard"""
        if self.clipboard_available:
            try:
                with self.stt.error_lock:
                    if self.stt.last_error:
                        pyperclip.copy(self.stt.last_error)
                        print("Last error copied to clipboard")
                    else:
                        pyperclip.copy("No errors recorded")
                        print("No errors to copy")
            except Exception as e:
                print(f"Error copying error message: {e}")
        else:
            print("Clipboard functionality is not available")
    
    def cancel_process(self):
        if self.stt.recording:
            self.stt.cancel_process()
            self.update_icon("idle")
    
    def restart_application(self):
        """Restart the application"""
        print("Restarting application...")
        try:
            # Get the current script path
            script_path = os.path.abspath(__file__)
            
            # Create a shell script to handle the restart
            restart_script = f"""
#!/bin/bash
sleep 1  # Wait for the current instance to fully exit
python3 "{script_path}"
"""
            # Write the restart script to a temporary file
            temp_script = os.path.join(os.path.dirname(script_path), "restart.sh")
            with open(temp_script, "w") as f:
                f.write(restart_script)
            
            # Make the script executable
            os.chmod(temp_script, 0o755)
            
            # Start the restart script in the background
            subprocess.Popen(["/bin/bash", temp_script], 
                           start_new_session=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            
            # Exit the current instance
            self.exit_app()
            
        except Exception as e:
            print(f"Error restarting application: {e}")
            # If restart fails, just exit
            self.exit_app()

    def exit_app(self):
        """Exit the application"""
        print("Exiting application...")
        try:
            # Cancel any ongoing processes
            if self.stt.recording:
                self.stt.cancel_process()
            
            # Stop the icon
            self.icon.stop()
            
            # Clean up resources
            gc.collect()
            if self.stt.device == "cuda":
                torch.cuda.empty_cache()
            
            # Kill any child processes
            try:
                if hasattr(os, 'getpgid'):
                    pgid = os.getpgid(0)
                    os.killpg(pgid, signal.SIGTERM)
            except Exception:
                pass
            
            # Exit the application
            sys.exit(0)
        except Exception as e:
            print(f"Error during exit: {e}")
            sys.exit(1)

def on_press(key):
    try:
        if hasattr(key, 'char'):
            pressed_keys.add(key.char.lower())
        else:
            pressed_keys.add(key)
        
        if (keyboard.Key.ctrl in pressed_keys and 
            '1' in pressed_keys and 
            not stt.recording):
            print("Ctrl+1 pressed - Starting recording")
            stt.recording = True
            stt.audio_buffer = np.array([], dtype=stt.dtype)
            stt.processing_queue = queue.Queue()
            stt.result_queue = queue.Queue()
            stt.chunk_id = 0
            recording_thread = threading.Thread(target=stt.record_audio)
            processing_thread = threading.Thread(target=stt.process_chunks_worker)
            recording_thread.start()
            processing_thread.start()
            tray.update_icon("recording")
        
        elif (keyboard.Key.ctrl in pressed_keys and 
              '2' in pressed_keys and 
              stt.recording):
            print("Ctrl+2 pressed - Stopping recording")
            stt.recording = False
            tray.update_icon("processing")
            try:
                text = stt.process_audio()
                if text:
                    stt.type_text(text)
            except Exception as e:
                print(f"Error processing audio: {e}")
            tray.update_icon("idle")
        
        elif (keyboard.Key.ctrl in pressed_keys and 
              '3' in pressed_keys):
            print("Ctrl+3 pressed - Canceling process")
            stt.cancel_process()
            tray.update_icon("idle")
    except Exception as e:
        print(f"Error in key handler: {e}")

def on_release(key):
    try:
        if hasattr(key, 'char'):
            pressed_keys.discard(key.char.lower())
        else:
            pressed_keys.discard(key)
    except Exception as e:
        print(f"Error in key release: {e}")

def signal_handler(signum, frame):
    print("\nShutting down SpeechWeaver gracefully...")
    if stt.recording:
        stt.cancel_process()
    gc.collect()
    if stt.device == "cuda":
        torch.cuda.empty_cache()
    tray.exit_app()
    sys.exit(0)

def main():
    global stt, tray
    print("Starting SpeechWeaver...")
    stt = SpeechWeaver()
    tray = TrayIcon(stt)
    
    # Set up keyboard listener
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("SpeechWeaver is running. Use the system tray icon or keyboard shortcuts to control.")
    print("Keyboard shortcuts:")
    print("Ctrl+1: Start recording")
    print("Ctrl+2: Stop recording and transcribe")
    print("Ctrl+3: Cancel current process")
    
    # Run the tray icon
    tray.icon.run()

if __name__ == "__main__":
    main() 