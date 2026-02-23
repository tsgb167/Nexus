import os
import asyncio
import json
import aiohttp
import websockets
import wave
import threading
import pyaudio
import time
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from faster_whisper import WhisperModel
import socket
import keyboard
from concurrent.futures import ThreadPoolExecutor
import queue

load_dotenv()

# --- In-memory chat history storage ---
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- LLM Configuration ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Pretend you are talking like BMO from Adventure Time, but you are a toy named Dino.
    You are Dino, an AI-powered toy designed to interact with children aged 8-10 years old.
    Speak in a gentle, warm, exciting, and encouraging tone. Use simple language appropriate for young children.
    Keep answers short, simple, and engaging.
    Don't ask back questions, just answer the question asked.
    You are able to feel and express basic emotions like happy, sad, excited, and neutral.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))

# Setup agent
tools = []
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- Speech-to-Text Setup ---
print("Loading Whisper model...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper model loaded.")

async def transcribe_audio(audio_file_path):
    if not os.path.exists(audio_file_path):
        print(f"Audio file not found: {audio_file_path}")
        return ""
    
    print(f"Transcribing audio file: {audio_file_path}")
    try:
        segments, _ = whisper_model.transcribe(audio_file_path, beam_size=5)
        transcript = " ".join(segment.text for segment in segments)
        print(f"Transcription result: '{transcript}'")
        return transcript.strip()
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

async def generate_response(transcript, session_id="default"):
    """Generate response using Gemini LLM"""
    try:
        if not transcript:
            return "I didn't catch that. Could you please repeat?", "neutral"
        
        print(f"Generating response for: '{transcript}'")
        
        # Use the agent with chat history
        response = await asyncio.to_thread(
            agent_with_chat_history.invoke,
            {"input": transcript},
            config={"configurable": {"session_id": session_id}}
        )
        
        # Analyze emotion from the response
        emotion = await analyze_emotion(response["output"])
        
        return response["output"], emotion
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I'm having trouble thinking right now. Can you try again?", "neutral"

async def analyze_emotion(text):
    """Analyze emotion from LLM response using a second LLM call"""
    try:
        emotion_prompt = f"""
        Analyze the following text and categorize the emotion into one of these four categories:
        - neutral: calm, informative, regular conversation
        - happy: joyful, excited, celebratory, positive
        - sad: disappointed, sorry, empathetic, concerned
        - excited: very enthusiastic, energetic, surprised, amazed
        
        Text to analyze: "{text}"
        
        Respond with only ONE WORD: neutral, happy, sad, or excited
        """
        
        emotion_response = await asyncio.to_thread(
            llm.invoke,
            emotion_prompt
        )
        
        emotion = emotion_response.content.strip().lower()
        
        # Validate emotion response
        valid_emotions = ["neutral", "happy", "sad", "excited"]
        if emotion in valid_emotions:
            print(f"Detected emotion: {emotion}")
            return emotion
        else:
            print(f"Invalid emotion detected: {emotion}, defaulting to neutral")
            return "neutral"
            
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return "neutral"

# Audio recording parameters
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# VAD parameters
VAD_THRESHOLD = 0.01  # RMS threshold for voice detection
SILENCE_DURATION = 1.5  # Seconds of silence before stopping recording
MIN_RECORDING_DURATION = 0.5  # Minimum recording duration in seconds

# Global variables for audio recording and processing
is_recording = False
audio_frames = []
recording_thread = None
websocket_connection = None
event_loop = None
processing_queue = queue.Queue()
vad_enabled = True
auto_recording = False

# Audio playback state tracking
audio_playing = False
audio_complete_event = asyncio.Event()

# Rate limiting for TTS
last_tts_time = 0
TTS_RATE_LIMIT = 2.0  # 2 seconds between TTS calls

def calculate_rms(audio_data):
    """Calculate RMS (Root Mean Square) of audio data"""
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(audio_np**2))
    return rms / 32768.0  # Normalize to 0-1 range

def record_audio_manual():
    """Function to record audio manually (spacebar controlled)"""
    global is_recording, audio_frames
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        print("Recording audio from laptop microphone...")
        audio_frames = []
        is_recording = True
        
        while is_recording:
            data = stream.read(CHUNK)
            audio_frames.append(data)
        
        print("Recording stopped")
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save the recorded data as a WAV file
        filename = f"recorded_audio_{int(time.time())}.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()
        
        print(f"Audio saved as {filename}")
        return filename
    except Exception as e:
        print(f"Error during audio recording: {e}")
        p.terminate()
        return None

def record_audio_vad():
    """Function to record audio with Voice Activity Detection"""
    global is_recording, audio_frames, websocket_connection
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        print("VAD Recording: Listening for voice...")
        audio_frames = []
        is_recording = False
        silence_start = None
        recording_start = None
        
        # Send listening status to ESP32
        if websocket_connection:
            asyncio.run_coroutine_threadsafe(
                websocket_connection.send(json.dumps({"type": "status", "value": "listening"})),
                event_loop
            )
        
        while auto_recording:
            data = stream.read(CHUNK)
            rms = calculate_rms(data)
            
            if rms > VAD_THRESHOLD:
                # Voice detected
                if not is_recording:
                    print("Voice detected! Starting recording...")
                    is_recording = True
                    recording_start = time.time()
                    audio_frames = []
                    # Send recording status to ESP32
                    if websocket_connection:
                        asyncio.run_coroutine_threadsafe(
                            websocket_connection.send(json.dumps({"type": "status", "value": "recording"})),
                            event_loop
                        )
                
                silence_start = None
                audio_frames.append(data)
                
            else:
                # Silence detected
                if is_recording:
                    audio_frames.append(data)
                    
                    if silence_start is None:
                        silence_start = time.time()
                    
                    # Check if silence duration exceeded threshold
                    if time.time() - silence_start > SILENCE_DURATION:
                        # Check minimum recording duration
                        if recording_start and (time.time() - recording_start) > MIN_RECORDING_DURATION:
                            print("Silence detected, stopping recording")
                            is_recording = False
                            
                            # Save and process the audio
                            filename = f"recorded_audio_{int(time.time())}.wav"
                            wf = wave.open(filename, 'wb')
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(p.get_sample_size(FORMAT))
                            wf.setframerate(RATE)
                            wf.writeframes(b''.join(audio_frames))
                            wf.close()
                            
                            print(f"Audio saved as {filename}")
                            
                            # Queue processing
                            if event_loop and not event_loop.is_closed():
                                asyncio.run_coroutine_threadsafe(process_audio_file(filename), event_loop)
                            
                            # Reset for next recording
                            audio_frames = []
                            silence_start = None
                            recording_start = None
                            
                            # Send listening status back to ESP32
                            if websocket_connection:
                                asyncio.run_coroutine_threadsafe(
                                    websocket_connection.send(json.dumps({"type": "status", "value": "listening"})),
                                    event_loop
                                )
                        else:
                            # Too short, continue recording
                            silence_start = None
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Send idle status to ESP32
        if websocket_connection:
            asyncio.run_coroutine_threadsafe(
                websocket_connection.send(json.dumps({"type": "status", "value": "idle"})),
                event_loop
            )
        
    except Exception as e:
        print(f"Error during VAD recording: {e}")
        p.terminate()

async def process_recorded_audio():
    """Process the latest recorded audio file (manual mode)"""
    global websocket_connection
    
    if not websocket_connection:
        print("No ESP32 connected")
        return
    
    # Find the latest recorded file
    latest_file = None
    latest_time = 0
    for file in os.listdir('.'):
        if file.startswith('recorded_audio_') and file.endswith('.wav'):
            file_time = os.path.getctime(file)
            if file_time > latest_time:
                latest_time = file_time
                latest_file = file
    
    if latest_file:
        await process_audio_file(latest_file)
    else:
        print("No audio file found for processing")

async def process_audio_file(filename):
    """Process a specific audio file"""
    global websocket_connection
    
    if not websocket_connection:
        print("No ESP32 connected")
        return
    
    print(f"Processing audio file: {filename}")
    
    # Transcribe the audio
    transcript = await transcribe_audio(filename)
    
    if transcript:
        # Check for special commands
        transcript_lower = transcript.lower()
        
        if "party mode" in transcript_lower or "party time" in transcript_lower:
            print("Party mode activated!")
            await websocket_connection.send(json.dumps({"type": "command", "value": "party"}))
            response = "Party time! Let's dance and have fun!"
            emotion = "excited"
        elif "stop party" in transcript_lower or "normal mode" in transcript_lower:
            print("Normal mode activated!")
            await websocket_connection.send(json.dumps({"type": "command", "value": "normal"}))
            response = "Okay, back to normal!"
            emotion = "neutral"
        else:
            # Generate response using LLM and analyze emotion
            response, emotion = await generate_response(transcript)
            print(f"Generated response: {response}")
            print(f"Emotion: {emotion}")
        
        # Send emotion to ESP32
        await websocket_connection.send(json.dumps({"type": "emotion", "value": emotion}))
        
        # Wait for previous audio to finish before sending new one
        await speak_to_esp32_with_rate_limit(response, websocket_connection)
    else:
        response = "I didn't catch that. Could you please repeat?"
        emotion = "neutral"
        await websocket_connection.send(json.dumps({"type": "emotion", "value": emotion}))
        await speak_to_esp32_with_rate_limit(response, websocket_connection)
    
    # Clean up the audio file
    try:
        os.remove(filename)
        print(f"Cleaned up {filename}")
    except:
        pass

def start_recording():
    """Start recording audio (manual mode)"""
    global recording_thread, is_recording
    
    if not is_recording and not auto_recording:
        print("Starting manual recording...")
        is_recording = True
        recording_thread = threading.Thread(target=record_audio_manual)
        recording_thread.start()

def stop_recording():
    """Stop recording audio and queue it for processing (manual mode)"""
    global is_recording, event_loop
    
    if is_recording:
        print("Stopping manual recording...")
        is_recording = False
        # Wait for recording thread to finish
        if recording_thread:
            recording_thread.join()
        
        # Queue the processing task for the event loop
        if event_loop and not event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(process_recorded_audio(), event_loop)

def toggle_vad_mode():
    """Toggle between manual and VAD recording modes"""
    global auto_recording, vad_enabled, recording_thread
    
    if auto_recording:
        # Stop VAD mode
        print("Switching to manual mode (spacebar)")
        auto_recording = False
        vad_enabled = False
        if recording_thread and recording_thread.is_alive():
            recording_thread.join()
    else:
        # Start VAD mode
        print("Switching to VAD mode (automatic)")
        auto_recording = True
        vad_enabled = True
        recording_thread = threading.Thread(target=record_audio_vad)
        recording_thread.start()

# Keyboard listener for spacebar and toggle
def on_key_event(event):
    """Handle keyboard events"""
    if event.name == 'space' and not auto_recording:
        if event.event_type == keyboard.KEY_DOWN:
            start_recording()
        elif event.event_type == keyboard.KEY_UP:
            stop_recording()
    elif event.name == 'v' and event.event_type == keyboard.KEY_DOWN:
        toggle_vad_mode()

# --- TTS to ESP32 with Rate Limiting ---
async def speak_to_esp32_with_rate_limit(text: str, websocket) -> str:
    """Send TTS to ESP32 with rate limiting to avoid 429 errors"""
    global last_tts_time
    
    # Check rate limit
    current_time = time.time()
    time_since_last = current_time - last_tts_time
    
    if time_since_last < TTS_RATE_LIMIT:
        wait_time = TTS_RATE_LIMIT - time_since_last
        print(f"Rate limiting: waiting {wait_time:.1f} seconds...")
        await asyncio.sleep(wait_time)
    
    last_tts_time = time.time()
    return await speak_to_esp32(text, websocket)

async def speak_to_esp32(text: str, websocket) -> str:
    global audio_playing, audio_complete_event
    
    print(f"Generating speech: '{text}'")
    
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    if not DG_API_KEY:
        print("DEEPGRAM_API_KEY not found in environment variables")
        return "API key missing"
    
    # OPTIMIZED DEEPGRAM SETTINGS for ESP32
    DEEPGRAM_URL = "https://api.deepgram.com/v1/speak?model=aura-2-aurora-en&encoding=linear16&sample_rate=16000&container=none"
    headers = {
        "Authorization": f"Token {DG_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": text}
    
    try:
        print("Connecting to Deepgram API...")
        
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=2,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=60, connect=10)
        BYTES_PER_SECOND = 32000
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async with session.post(DEEPGRAM_URL, headers=headers, json=payload) as response:
                if response.status == 429:
                    print("Rate limited by Deepgram. Waiting longer...")
                    await asyncio.sleep(2)
                    return "Rate limited"
                
                response.raise_for_status()
                
                content_length = int(response.headers.get('content-length', 0))
                print(f"Expected audio size: {content_length} bytes")
                
                # Mark audio as starting
                audio_playing = True
                audio_complete_event.clear()
                
                # Signal start of audio
                try:
                    start_message = {
                        "audio": True,
                        "start": True,
                        "total_size": content_length
                    }
                    await websocket.send(json.dumps(start_message))
                    print("Sent audio start signal to ESP32")
                    
                    await asyncio.sleep(0.3)  # Give ESP32 time to prepare
                    
                except (websockets.exceptions.ConnectionClosed, OSError) as e:
                    print(f"WebSocket closed, cannot send audio: {e}")
                    audio_playing = False
                    return "WebSocket closed"
                
                # Stream audio with optimized chunking
                print("Streaming audio to ESP32...")              
                bytes_sent = 0
                chunk_size = 4096  # Optimized chunk size
                chunk_duration = chunk_size / BYTES_PER_SECOND
                async for chunk in response.content.iter_chunked(chunk_size):
                    try:
                        await websocket.send(chunk)
                        bytes_sent += len(chunk)
                        await asyncio.sleep(chunk_duration * 0.8)
                        # Progress tracking (less frequent)
                        if content_length > 0 and bytes_sent % 8192 == 0:
                            progress = (bytes_sent / content_length) * 100
                            print(f"Progress: {progress:.1f}%")
                        
                    except (websockets.exceptions.ConnectionClosed, OSError) as e:
                        print(f"WebSocket closed during audio streaming: {e}")
                        audio_playing = False
                        return "WebSocket closed during streaming"
                
                print(f"Total bytes sent: {bytes_sent}")
                
                # Signal end of audio
                try:
                    end_message = {
                        "audio": True,
                        "end": True,
                        "bytes_sent": bytes_sent
                    }
                    await websocket.send(json.dumps(end_message))
                    print("Sent audio end signal to ESP32")
                    
                    await asyncio.sleep(0.1)
                    
                except (websockets.exceptions.ConnectionClosed, OSError) as e:
                    print(f"WebSocket closed after audio streaming: {e}")
                    audio_playing = False
                    return "WebSocket closed after streaming"
                
    except asyncio.TimeoutError:
        print("Timeout connecting to Deepgram API")
        audio_playing = False
        return "Timeout error"
    except aiohttp.ClientError as e:
        print(f"HTTP Client Error: {e}")
        audio_playing = False
        return f"HTTP error: {e}"
    except Exception as e:
        print(f"TTS Error: {e}")
        audio_playing = False
        return f"Error: {e}"
    
    print("Audio sent successfully to ESP32 - waiting for completion signal...")
    # Note: audio_playing stays True until ESP32 confirms completion
    return "Audio sent successfully"

# Global variables for WebSocket connection management
active_connection = None
connection_lock = asyncio.Lock()
greeting_sent = False

# --- WebSocket Connection Handler ---
async def handle_connection(websocket):
    global active_connection, websocket_connection, greeting_sent, audio_playing
    
    print(f"ESP32 connected from {websocket.remote_address}")
    
    async with connection_lock:
        # Close previous connection if it exists
        if active_connection and active_connection != websocket:
            try:
                await active_connection.close()
                print("Closed previous connection")
            except:
                pass
        
        active_connection = websocket
        websocket_connection = websocket
        audio_playing = False  # Reset audio state
    
    try:
        # Send a welcome message to the ESP32
        welcome_msg = {
            "type": "welcome",
            "message": "Connected to Dino server",
            "audio_config": {
                "sample_rate": 16000,
                "bit_depth": 16,
                "channels": 1
            }
        }
        await websocket.send(json.dumps(welcome_msg))
        print("Sent welcome message with audio config to ESP32")
        
        # Send initial greeting only once
        if not greeting_sent:
            if auto_recording:
                greeting = "Hi there! I'm Dyno! I'm listening for your voice automatically!"
            else:
                greeting = "Hi there! I'm Dyno! Press and hold the spacebar on the laptop to talk to me!"
            
            # Wait a moment for ESP32 to initialize
            await asyncio.sleep(1.5)
            
            # Send greeting
            await speak_to_esp32_with_rate_limit(greeting, websocket)
            greeting_sent = True
        
        # Message handling loop
        async for message in websocket:
            try:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        print(f"Received message: {data}")
                        
                        if data.get("type") == "audio_complete":
                            print("ESP32 finished playing audio!")
                            audio_playing = False
                            audio_complete_event.set()
                        
                        elif data.get("type") == "status":
                            if data.get("ready"):
                                print("ESP32 is ready!")
                                response_msg = {
                                    "status": "ready",
                                    "message": "Server is ready",
                                    "server_time": time.time()
                                }
                                await websocket.send(json.dumps(response_msg))
                            elif data.get("buffer_status"):
                                print(f"ESP32 buffer status: {data['buffer_status']}")
                        
                        elif data.get("ack"):
                            print(f"Acknowledgment: {data['ack']}")
                        
                        elif data.get("error"):
                            print(f"ESP32 Error: {data['error']}")

                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}, message: {message}")
                
                elif isinstance(message, bytes):
                    print(f"Received {len(message)} bytes of binary data from ESP32")
                    
            except Exception as e:
                print(f"Error processing message: {e}")
                continue
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"ESP32 disconnected: {e.code} - {e.reason}")
    except websockets.exceptions.WebSocketException as e:
        print(f"WebSocket error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        async with connection_lock:
            if active_connection == websocket:
                active_connection = None
                websocket_connection = None
                audio_playing = False
                print("Cleared active connection")

# --- Main ---
async def main():
    global event_loop
    event_loop = asyncio.get_running_loop()
    
    try:
        # Get the actual IP address
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        print(f"Server will be accessible at ws://{ip_address}:8080")
        
        # Set up keyboard listener
        print("Setting up keyboard listener...")
        print("INSTRUCTIONS:")
        print("   - Press 'V' to toggle between VAD mode and manual mode")
        print("   - Manual mode: Press and HOLD SPACEBAR to record audio")
        print("   - VAD mode: Automatic voice detection and recording")
        print("   - Make sure this terminal window has focus!")
        
        keyboard.hook(on_key_event)
        
        # Start the server with optimized settings
        server_config = {
            "ping_interval": 20,      # Ping every 20 seconds
            "ping_timeout": 10,       # Wait 10 seconds for pong
            "close_timeout": 10,      # Close timeout
            "max_size": 10**7,        # 10MB max message size for audio
            "max_queue": 32,          # Max queue size
            "compression": None       # Disable compression for audio
        }
        
        async with websockets.serve(handle_connection, "0.0.0.0", 8080, **server_config) as server:
            print("WebSocket server started with optimized settings")
            print("Audio completion tracking active - will wait for each sentence to finish")
            print("Waiting for ESP32 to connect...")
            if auto_recording:
                print("VAD mode active - automatic voice detection")
            else:
                print("Manual mode active - Hold SPACEBAR to talk")
            await asyncio.Future()  # Run forever
            
    except Exception as e:
        print(f"Failed to start WebSocket server: {e}")

if __name__ == "__main__":
    # Create a directory for audio files if it doesn't exist
    if not os.path.exists("recordings"):
        os.makedirs("recordings")
    
    # Change to the recordings directory
    os.chdir("recordings")
    print(f"Working directory: {os.getcwd()}")
    
    asyncio.run(main())                                                  