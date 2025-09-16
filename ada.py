import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from groq import Groq
from PIL import ImageGrab, Image
from openai import OpenAI
from faster_whisper import WhisperModel
import speech_recognition as sr
import google.generativeai as genai
import pyperclip
import cv2
import pyaudio
import os
import threading
import time
import keyboard
import re
import json
from ultralytics import YOLO
import numpy as np
from collections import deque
segmentation_model = YOLO("yolo11m-seg.pt") 

wake_word = 'Lady Ada'
# Add camera selection in the global variable area at the beginning of the file
CAMERA_SOURCE = 0  # 0: built-in camera, 1: external webcam, 2: DroidCam

def load_config():
    with open('config.json') as f:
        return json.load(f)

config = load_config()
groq_client = Groq(api_key=config['groq_api_key'])
genai.configure(api_key=config['google_api_key'])
openai_client = OpenAI(api_key=config['openai_api_key'])
web_cam = cv2.VideoCapture(CAMERA_SOURCE)
web_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
web_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Add a global variable to control voice interaction
use_voice_interaction = False # Set this to False to disable voice interaction
stop_event = threading.Event()

# Add the following content in the global variable area
# Queue for storing conversation history
chat_history = deque(maxlen=5)  # Store the last 5 conversations
frame_buffer = None
overlay_frame = None

# Add in the global variable area
should_exit = False

# Add global variable for storing current segmentation results
current_results = None

# Add in the global variable area
video_writer = None
recording_started = False

# Add audio recording related variables in the global variable area
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
audio_frames = []
p_record = pyaudio.PyAudio()
recording_stream = None

# Add in the global variable area
current_video_path = None

# Add in the global variable area
is_speaking = False  # Used to track space key status

# Add in the global variable area
waiting_sound_thread = None
stop_waiting_sound = threading.Event()

# Add recording control flag in the global variable area
ENABLE_RECORDING = True  # Set to False to disable video and audio recording

# Add in the global variable area
continuous_segmentation = False  # Control whether to continuously segment

# Add in the global variable area
detection_buffer = deque(maxlen=5)  # Store detection results from the last 5 frames

sys_msg = (
    'You are a multi-modal AI voice assistant named Ada, after the British computer scientist Ada Lovelace,'
    ' Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed'
    'text prompt that will be attached to their transcribed voice prompt, Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response beforeadding new tokens to the response. '
    'Do not expect or request images, just use the context if added.Use all of the context of this conversation so your response is relevant to the conversation.'
    ' Make your responses clear and concise, avoiding any verbosity.'
)

CLASSES = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'airplane': 4,
    'bus': 5,
    'train': 6,
    'truck': 7,
    'boat': 8,
    'traffic light': 9,
    'fire hydrant': 10,
    'stop sign': 11,
    'parking meter': 12,
    'bench': 13,
    'bird': 14,
    'cat': 15,
    'dog': 16,
    'horse': 17,
    'sheep': 18,
    'cow': 19,
    'elephant': 20,
    'bear': 21,
    'zebra': 22,
    'giraffe': 23,
    'backpack': 24,
    'umbrella': 25,
    'handbag': 26,
    'tie': 27,
    'suitcase': 28,
    'frisbee': 29,
    'skis': 30,
    'snowboard': 31,
    'sports ball': 32,
    'kite': 33,
    'baseball bat': 34,
    'baseball glove': 35,
    'skateboard': 36,
    'surfboard': 37,
    'tennis racket': 38,
    'bottle': 39,
    'wine glass': 40,
    'cup': 41,
    'fork': 42,
    'knife': 43,
    'spoon': 44,
    'bowl': 45,
    'banana': 46,
    'apple': 47,
    'sandwich': 48,
    'orange': 49,
    'broccoli': 50,
    'carrot': 51,
    'hot dog': 52,
    'pizza': 53,
    'donut': 54,
    'cake': 55,
    'chair': 56,
    'couch': 57,
    'potted plant': 58,
    'bed': 59,
    'dining table': 60,
    'toilet': 61,
    'tv': 62,
    'laptop': 63,
    'mouse': 64,
    'remote': 65,
    'keyboard': 66,
    'cell phone': 67,
    'microwave': 68,
    'oven': 69,
    'toaster': 70,
    'sink': 71,
    'refrigerator': 72,
    'book': 73,
    'clock': 74,
    'vase': 75,
    'scissors': 76,
    'teddy bear': 77,
    'hair drier': 78,
    'toothbrush': 79
}
# Modify color mapping in the global variable area, using more unique colors
CLASS_COLORS = {
    class_id: color for class_id, color in enumerate([
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
        (255, 255, 0),   # Yellow
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cyan
        (128, 0, 0),     # Dark Red
        (0, 128, 0),     # Dark Green
        (0, 0, 128),     # Dark Blue
        (128, 128, 0),   # Olive
        (128, 0, 128),   # Purple
        (0, 128, 128),   # Teal
        (255, 128, 0),   # Orange
        (255, 0, 128),   # Pink
        (128, 255, 0),   # Lime
        (0, 255, 128),   # Spring Green
    ] * ((len(CLASSES) // 16) + 1))  # Repeat color list until covering all classes
}
convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048,
}

safety_settings = [
    {
    'category': 'HARM_CATEGORY_HARASSMENT',
    'threshold': 'BLOCK_NONE',
    },
    {
    'category': 'HARM_CATEGORY_HATE_SPEECH',
    'threshold': 'BLOCK_NONE',
    },
    {
    'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
    'threshold': 'BLOCK_NONE',
    },
    {
    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
    'threshold': 'BLOCK_NONE',
    },
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              safety_settings=safety_settings,
                              generation_config=generation_config)

num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device = 'cpu',
    compute_type = 'int8',
    cpu_threads = num_cores // 2,
    num_workers = num_cores // 2,
)

r = sr.Recognizer()
source = sr.Microphone()

def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n  IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama-3.1-8b-instant')
    response = chat_completion.choices[0].message
    convo.append(response)
    
    return response.content
    
# Add intent matching function
def detect_intent(prompt):
    """Use Groq API to detect user intent"""
    sys_msg = ('''
    You are an intent classification AI. Analyze the user's input and determine which of the following intents it most closely matches:
    1. find_objects - User wants to find or locate specific objects in their view
    2. describe_scene - User wants a description of what's currently visible
    3. count_objects - User wants to count specific objects
    4. position_query - User wants to know where specific objects are located
    5. take_screenshot - User wants to capture a screenshot
    6. clipboard_extract - User wants to extract text from clipboard
    7. quit_request - User wants to exit the program
    8. general_chat - None of the above, user just wants a normal conversation
    9. real-time segmentation - User wants to segment all objects in real-time
    10. visual_question - User is asking a specific question about what they can see
    Respond ONLY with the intent name (e.g., "find_objects", "general_chat", etc.) without any explanation.
    ''')
    
    intent_convo = [
        {'role': 'system', 'content': sys_msg},
        {'role': 'user', 'content': prompt}
    ]
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=intent_convo, 
            model='llama-3.1-8b-instant',
            temperature=0.1,  # Low temperature for more deterministic results
            max_tokens=10     # Only need short answers
        )
        intent = chat_completion.choices[0].message.content.strip().lower()
        print(f"Detected intent: {intent}")
        return intent
    except Exception as e:
        print(f"Error detecting intent: {e}")
        return "general_chat"  # Default to general chat

# Modify translate_user_classes function, optimize multi-category processing
def translate_user_classes(prompt):
    """
    Translate non-standard category names input by the user into YOLO-supported standard categories.
    For example: `"men and women"` → `"person"`, `"laptop computer"` → `"laptop"`
    Support multi-category recognition, e.g., `"find people and cars"` → `["person", "car"]`
    Support generic label mapping, e.g., `"find fruit"` → `["apple", "orange", "banana"]`

    """
    # First check generic label mapping
    try:
        with open('category_labels.json', 'r') as f:
            category_labels = json.load(f)
        
        prompt_lower = prompt.lower()
        expanded_classes = []
        
        # Check if user input contains generic labels
        for label, classes in category_labels.items():
            if label in prompt_lower:
                expanded_classes.extend(classes)
        
        # If generic label mapping is found, return directly
        if expanded_classes:
            # Remove duplicates and verify categories are in CLASSES
            valid_classes = []
            for cls in expanded_classes:
                if cls in CLASSES and cls not in valid_classes:
                    valid_classes.append(cls)
            
            if valid_classes:
                print(f"Found category label mapping: {valid_classes}")
                return valid_classes
    except Exception as e:
        print(f"Error loading category labels: {e}")
    
    # If no generic label mapping is found, continue with existing AI translation logic
    # Build system prompt
    classes_list = ", ".join(CLASSES.keys())
    sys_prompt = f"""
    You are a computer vision assistant. Your task is to map user's natural language descriptions 
    to the standard YOLO object detection classes.

    Available YOLO classes: {classes_list}

    Rules:
    1. Map user terms to the closest matching YOLO class(es)
    2. Consider synonyms, plurals, and related terms
    3. Return ONLY the matching class name(s) without any explanation
    4. If multiple classes match, return them separated by commas
    5. If no classes match, return "NONE"
    6. Be comprehensive - identify ALL relevant classes mentioned in the query

    Examples:
    - "men and women" -> "person"
    - "laptop computer" -> "laptop"
    - "dining table with food" -> "dining table"
    - "coca cola bottle" -> "bottle"
    - "mobile phone" -> "cell phone"
    - "golden retriever" -> "dog"
    - "unicorn" -> "NONE"
    - "find people and cars" -> "person, car"
    - "look for cats and dogs" -> "cat, dog"
    """

    user_prompt = f"User query: {prompt}\nMatching YOLO classes:"
    
    # Call Groq API for analysis
    try:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.1,  # Low temperature for more deterministic results
            max_tokens=50
        )
        
        # Parse response
        mapped_classes = response.choices[0].message.content.strip()
        
        # If no matching categories
        if mapped_classes.upper() == "NONE":
            return None
        
        # Split multiple categories
        class_list = [cls.strip() for cls in mapped_classes.split(',')]
        
        # Verify returned categories are in CLASSES
        valid_classes = []
        for cls in class_list:
            if cls in CLASSES:
                valid_classes.append(cls)
        
        return valid_classes if valid_classes else None
    
    except Exception as e:
        print(f"Error translating user classes: {e}")
        return None

# Modify parse_target_classes function, integrate category translation functionality
def parse_target_classes(prompt):
    """Extract target class IDs from user prompt with translation support"""
    # First try direct matching
    target_classes = []
    prompt = prompt.lower()
    
    # Check each class name in the prompt
    for class_name, class_id in CLASSES.items():
        if class_name in prompt:
            target_classes.append(class_id)
    
    # If no direct match found, try using translation
    if not target_classes:
        translated_classes = translate_user_classes(prompt)
        if translated_classes:
            for class_name in translated_classes:
                class_id = CLASSES.get(class_name)
                if class_id is not None:
                    target_classes.append(class_id)
    
    # If categories are found, return them, otherwise return None
    return target_classes if target_classes else None

# Modify function_call function's real-time segmentation part, add handling for unsupported categories
def function_call(prompt):
    # Keep existing direct matching logic
    # Check if it's a describe frame request
    description_keywords = ['describe frame', 'what do you see', 'analyze scene', 'describe scene']
    prompt_lower = prompt.lower()
    
    # If contains description keywords
    if any(keyword in prompt_lower for keyword in description_keywords):
        return "describe frame"
    
    # Add position query detection
    location_keywords = ['where', 'location', 'position']
    
    # Check if it's a search request
    if 'find' in prompt_lower and 'for me' in prompt_lower:
        # Check if requesting to find unsupported objects
        translated_classes = translate_user_classes(prompt)
        if translated_classes is None:
            # This is an unsupported segmentation object, return visual question
            return "visual question"
        return "real-time segmentation"
    
    # Rest of the code remains unchanged
    if any(keyword in prompt_lower for keyword in location_keywords):
        for class_name in CLASSES.keys():
            if class_name in prompt_lower:
                return "position query"
    
    # Add some keywords to detect counting queries
    counting_keywords = ['how many', 'count', 'number of']
    
    # Check if it's a counting query
    if any(keyword in prompt_lower for keyword in counting_keywords):
        for class_name in CLASSES.keys():
            if class_name in prompt_lower:
                return "count objects"
    
    # Use AI intent detection
    intent = detect_intent(prompt)
    
    # Return corresponding function based on detected intent
    if intent == "find_objects":
        # Check if requesting to find unsupported objects
        translated_classes = translate_user_classes(prompt)
        if translated_classes is None:
            # This is an unsupported segmentation object, return visual question
            return "visual question"
        return "real-time segmentation"
    elif intent == "describe_scene":
        return "describe frame"
    elif intent == "visual_question":
        return "visual question"
    elif intent == "count_objects":
        return "count objects"
    elif intent == "position_query":
        return "position query"
    elif intent == "take_screenshot":
        return "take screenshot"
    elif intent == "clipboard_extract":
        return "extract clipboard"
    elif intent == "quit_request":
        return "quit"
    else:  # general_chat or any other case
        # Fallback to existing function call logic
        sys_msg = (
            'You are an AI function calling model. You will determine whether extracting the users clipboard content, '
            'taking a screenshot, calling no functions is best for a voice assistant to respond '
            'to the users prompt, The webcam can be assumed to be a normal laptop webcam facing the user. You will '
            'respond with only one selection from this list: ["extract clipboard", "real-time segmentation", "well done",'
            '"take screenshot", "count objects", "position query", "quit", "None"] \n'
            'Do not respond with anything but the most logical selection from that list with no explanations. Format the'
            'function call name exactly as I listed.'
        )
        
        function_convo = [{'role': 'system', 'content': sys_msg},
                          {'role': 'user', 'content': prompt}]
                          
        chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
        response = chat_completion.choices[0].message
        
        return response.content

def take_screenshot():
    path = 'screenshot.png'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality = 15)
    
def get_clipboard():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        return 'Error: Could not get clipboard content'
    
def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semtantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI'
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text
    
def play_waiting_sound():
    """Play waiting sound effect"""
    try:
        import winsound
        winsound.PlaySound('./detector.wav', winsound.SND_ASYNC | winsound.SND_LOOP)
        
        while not stop_waiting_sound.is_set():
            time.sleep(0.1)
        
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception as e:
        print(f"Error playing waiting sound: {e}")

def start_waiting_sound():
    """Start playing waiting sound effect"""
    try:
        global waiting_sound_thread, stop_waiting_sound
        stop_waiting_sound.clear()
        waiting_sound_thread = threading.Thread(target=play_waiting_sound)
        waiting_sound_thread.daemon = True
        waiting_sound_thread.start()
    except Exception as e:
        print(f"Error starting waiting sound: {e}")

def stop_waiting_sound_thread():
    """Stop waiting sound effect"""
    try:
        global stop_waiting_sound
        if stop_waiting_sound is not None:
            stop_waiting_sound.set()
            time.sleep(0.2)  # Give some time for the sound to stop
    except Exception as e:
        print(f"Error stopping waiting sound: {e}")
    
def speak(text):
    """Modify speak function, decide whether to generate speech based on voice_interaction flag"""
    chat_history.append(f"Ada: {text}")  # Add to history record
    
    # Stop waiting sound effect
    stop_waiting_sound_thread()
    
    # If voice interaction is disabled, only display text without generating speech
    if not use_voice_interaction:
        return
    
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False
    
    with openai_client.audio.speech.with_streaming_response.create(
        model = 'tts-1',
        voice = 'shimmer',
        response_format = 'pcm',
        input = text,
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                amplitude = max(chunk)
                if amplitude > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text
    
# Function to handle segmentation in a separate thread
def segmentation_thread(prompt):
    """Execute segmentation in a separate thread"""
    try:
        result = start_segmentation(prompt)
        if result is None or (hasattr(result, 'boxes') and len(result.boxes) == 0):
            print("No objects were detected during segmentation.")
            speak("I couldn't find any of the objects you were looking for.")
    except Exception as e:
        # Simplify error messages to avoid excessive repeated output
        if "object of type 'NoneType' has no len()" in str(e):
            # Silently handle common cases of missing detected objects
            print("No objects detected in this frame.")
        else:
            print(f"Segmentation error: {e}")
    
def get_class_names(target_classes):
    """Convert class IDs to names"""
    names = []
    for class_id in target_classes:
        for name, id in CLASSES.items():
            if id == class_id:
                names.append(name)
                break
    return names
    
def add_text_to_frame(frame, text_lines):
    """Add text to the bottom left corner of the frame, distinguish between user and Ada's speech"""
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    padding = 10
    line_spacing = 30
    
    # Calculate starting y coordinate (from bottom up)
    y = height - padding - (len(text_lines) - 1) * line_spacing
    
    for line in text_lines:
        # Add black background
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(frame, 
                     (padding, int(y - text_height)), 
                     (padding + text_width, int(y + 5)), 
                     (0, 0, 0), 
                     -1)
        
        # Choose different text colors based on speaker
        if line.startswith("User:"):
            text_color = (255, 200, 0)  # User speech in golden yellow
        else:
            text_color = (255, 255, 255)  # Ada speech in white
            
        cv2.putText(frame, line, 
                    (padding, int(y)), 
                    font, 
                    font_scale, 
                    text_color, 
                    thickness)
        y += line_spacing
    return frame

def setup_video_writer():
    """Setup video writer"""
    global video_writer, current_video_path
    
    # If recording is disabled, return directly
    if not ENABLE_RECORDING:
        return None
        
    # Ensure recordings directory exists
    os.makedirs('../runs/recordings', exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_video_path = f'../runs/recordings/recording_{timestamp}.mp4'
    
    # Get video parameters
    width = int(web_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(web_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(current_video_path, fourcc, fps, (width, height))
    return current_video_path

def display_video_thread():
    """Thread function for continuously displaying video stream"""
    global frame_buffer, overlay_frame, should_exit, recording_started, video_writer
    
    cv2.namedWindow('Ada Video Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Ada Video Feed', 1280, 720)
    
    # Initialize video writer
    if not recording_started and ENABLE_RECORDING:
        output_path = setup_video_writer()
        print(f"Recording to: {output_path}")
        recording_started = True
    elif not ENABLE_RECORDING:
        print("Video recording is disabled.")
    
    while not should_exit:
        ret, frame = web_cam.read()
        if not ret:
            continue
            
        frame_buffer = frame.copy()
        display_frame = frame.copy()
        
        # If there are segmentation results, overlay display
        if overlay_frame is not None:
            display_frame = cv2.addWeighted(display_frame, 0.7, overlay_frame, 0.3, 0)
            
        # Add conversation history
        display_frame = add_text_to_frame(display_frame, list(chat_history))
        
        # Write frame to video file
        if video_writer is not None and ENABLE_RECORDING:
            video_writer.write(display_frame)
        
        cv2.imshow('Ada Video Feed', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up resources
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    web_cam.release()
    
# Modify process_segmentation_results function, ensure proper handling of multi-category masks
def process_segmentation_results(results, frame):
    """Visualize segmentation results to overlay layer, support multi-category simultaneous display"""
    if results is None:
        return None
        
    # Create transparent overlay layer
    overlay = np.zeros_like(frame)
    
    height, width = frame.shape[:2]
    
    # Process each detected object
    try:
        # Process segmentation masks (if any) - process masks first then bounding boxes, avoid bounding boxes being covered by masks
        if hasattr(results, 'masks') and results.masks is not None:
            for i, mask in enumerate(results.masks.data):
                class_id = int(results.boxes.cls[i])
                color = CLASS_COLORS[class_id]
                
                # Convert mask to numpy array and resize
                mask_np = mask.cpu().numpy()
                mask_np = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_LINEAR)
                mask_np = mask_np > 0.5  # Binarization
                
                # Apply mask and maintain color distinction between different categories
                alpha = 0.4  # Reduce transparency to allow multiple categories to be distinguished
                overlay[mask_np] = (overlay[mask_np] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        
        # Process bounding boxes
        for i, box in enumerate(results.boxes.xyxy):
            # Get class ID and name
            class_id = int(results.boxes.cls[i])
            conf = float(results.boxes.conf[i])  # Get confidence
            
            # Get corresponding class name
            class_name = None
            for name, id in CLASSES.items():
                if id == class_id:
                    class_name = name
                    break
                    
            if class_name is None:
                continue
                
            # Set color
            color = CLASS_COLORS[class_id]
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Add confidence text
            conf_text = f"{class_name}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                conf_text, font, font_scale, thickness
            )
            
            # Text position (above box)
            text_x = x1
            text_y = y1 - 5
            
            # Ensure text is within image
            if text_y < text_height:
                text_y = y1 + text_height + 5
                
            # Draw text background
            cv2.rectangle(
                overlay, 
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                color, 
                -1
            )
            
            # Draw text
            cv2.putText(
                overlay,
                conf_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )
    
    except Exception as e:
        print(f"Error in processing segmentation results: {e}")
    
    return overlay

# Add function to get best detection result (frame with most objects)
def get_best_detection():
    """Get the result with the most object detections from the buffer"""
    global detection_buffer
    
    if not detection_buffer:
        return None
        
    # Find the result containing the most objects
    max_objects = 0
    best_result = None
    
    for result in detection_buffer:
        if result is not None and hasattr(result, 'boxes'):
            num_objects = len(result.boxes)
            if num_objects > max_objects:
                max_objects = num_objects
                best_result = result
                
    return best_result

# Modify start_segmentation function, optimize multi-category processing
def start_segmentation(prompt):
    """Start object segmentation, support multi-category detection"""
    global segmentation_model, segmentation_mode, current_results, stop_event, overlay_frame
    global continuous_segmentation, detection_buffer
    
    # Clear detection buffer
    detection_buffer.clear()
    
    # Reset stop event
    stop_event.clear()
    
    # Enable continuous segmentation mode
    continuous_segmentation = True

    # Parse target categories
    target_classes = parse_target_classes(prompt)
    
    # If no specific target category is specified, use all categories
    if target_classes is None:
        print(f"No specific object class detected in prompt. Showing all detectable objects.")
        target_classes = list(CLASSES.values())
    else:
        print(f"Target classes: {target_classes}")
        
    # Get target category names
    target_names = get_class_names(target_classes)
    print(f"Looking for: {', '.join(target_names)}")
    speak(f"Looking for {', '.join(target_names)}")
    
    segmentation_mode = True
    segmentation_start_time = time.time()
    first_detection = False
    
    try:
        # Continuously process frames until stop signal is received
        while not stop_event.is_set():
            if frame_buffer is None:
                time.sleep(0.1)
                continue
                
            # Process current frame
            try:
                # Add timeout check, stop if no objects found for a long time
                if time.time() - segmentation_start_time > 30 and not first_detection:
                    print("Segmentation timeout - no objects found within 30 seconds.")
                    speak("I couldn't find the objects you're looking for after searching for 30 seconds.")
                    break
                    
                # Prediction results
                results = segmentation_model.predict(
                    source=frame_buffer,
                    save=False,
                    show=False,
                    verbose=False,
                    conf=0.15,
                    classes=target_classes,
                    retina_masks=True
                )
                
                # Add result check
                if results is None or len(results) == 0:
                    # Handle case with no results
                    overlay_frame = None  # Clear overlay layer
                    detection_buffer.append(None)  # Add empty result to buffer
                    continue
                
                # Check if any objects were found
                if len(results[0].boxes) == 0:
                    # No objects found, clear overlay layer and continue to next frame
                    overlay_frame = None
                    detection_buffer.append(None)  # Add empty result to buffer
                    continue
                
                # Update current results
                current_results = results[0] if len(results) > 0 else None
                
                # Add current result to buffer
                if current_results is not None:
                    detection_buffer.append(current_results)
                
                # Process segmentation results and update overlay layer
                overlay_frame = process_segmentation_results(current_results, frame_buffer)
                
                # If objects are found, notify user (only first time)
                if current_results is not None and len(current_results.boxes) > 0 and not first_detection:
                    first_detection = True
                    
                    # Count the number of each detected category
                    found_classes = {}
                    for cls in current_results.boxes.cls:
                        class_id = int(cls)
                        class_name = [name for name, id in CLASSES.items() if id == class_id][0]
                        found_classes[class_name] = found_classes.get(class_name, 0) + 1
                    
                    # Build detection result message
                    detection_message = "I found "
                    class_details = []
                    for class_name, count in found_classes.items():
                        if count == 1:
                            class_details.append(f"a {class_name}")
                        else:
                            class_details.append(f"{count} {class_name}s")
                    
                    detection_message += ", ".join(class_details)
                    
                    print(f"Detection result: {detection_message}")
                    speak(detection_message)
                
                # If not continuous mode and objects have been found, exit loop
                if not continuous_segmentation and first_detection:
                    break
                
            except Exception as e:
                # Only print error messages for non-NoneType errors
                if "object of type 'NoneType' has no len()" not in str(e):
                    print(f"Segmentation processing error: {e}")
                # Continue processing next frame
                continue
                
            # Brief sleep to reduce CPU usage
            time.sleep(0.05)
            
    except Exception as e:
        # Catch other exceptions
        print(f"Segmentation error: {e}")
    finally:
        # End segmentation mode
        segmentation_mode = False
        if not continuous_segmentation:
            overlay_frame = None  # If not continuous mode, clear overlay layer
        return current_results

# Modify stop_segmentation function
def stop_segmentation():
    """Stop segmentation and clear display results"""
    global stop_event, overlay_frame, continuous_segmentation
    
    stop_event.set()  # Signal thread to stop
    continuous_segmentation = False  # Disable continuous segmentation mode
    overlay_frame = None  # Clear overlay layer
    speak("Stopping segmentation")

def get_instance_count(class_name):
    """Get the number of instances of a specific category in the current scene, using multi-frame buffering technique"""
    # Get best detection result (frame with most objects)
    best_result = get_best_detection()
    
    if best_result is None:
        return 0
        
    class_id = CLASSES.get(class_name.lower())
    if class_id is None:
        return 0
        
    # Count instances of specified category
    count = sum(1 for cls in best_result.boxes.cls if int(cls) == class_id)
    return count

def format_count_response(count, class_name):
    """Format response text based on count"""
    if count == 1:
        return f"I can see 1 {class_name} in the current scene."
    else:
        return f"I can see {count} {class_name}s in the current scene."

# Modify count query handling in callback and start_listening
def handle_count_query(clean_prompt):
    """Unified function for handling count queries, support multi-category queries"""
    prompt_lower = clean_prompt.lower()
    found_classes = []
    responses = []
    
    # Collect all categories mentioned in the prompt
    for class_name in CLASSES.keys():
        if class_name in prompt_lower:
            found_classes.append(class_name)
    
    if not found_classes:
        return False
        
    # Get count for each category and generate response
    for class_name in found_classes:
        count = get_instance_count(class_name)
        if count == 1:
            responses.append(f"1 {class_name}")
        else:
            responses.append(f"{count} {class_name}s")
    
    # Combine all responses
    if len(responses) == 1:
        final_response = f"I can see {responses[0]} in the current scene."
    elif len(responses) == 2:
        final_response = f"I can see {responses[0]} and {responses[1]} in the current scene."
    else:
        final_response = "I can see " + ", ".join(responses[:-1]) + f", and {responses[-1]} in the current scene."
    
    print(f'Ada: {final_response}')
    speak(final_response)
    return True

def get_region(x, y, width, height):
    """Determine which region of the screen a coordinate is in"""
    x_ratio = x / width
    y_ratio = y / height
    
    # Horizontal position
    if x_ratio < 0.33:
        h_pos = "left"
    elif x_ratio > 0.67:
        h_pos = "right"
    else:
        h_pos = "center"
        
    # Vertical position
    if y_ratio < 0.33:
        v_pos = "upper"
    elif y_ratio > 0.67:
        v_pos = "lower"
    else:
        v_pos = "middle"
        
    # Combine position
    if h_pos == "center" and v_pos == "middle":
        return "center"
    elif h_pos == "center":
        return f"{v_pos} {h_pos}"
    elif v_pos == "middle":
        return h_pos
    else:
        return f"{v_pos} {h_pos}"

def draw_arrow(frame, target_center, color=(255, 255, 255), thickness=3):
    """Draw arrow pointing to target on frame"""
    height, width = frame.shape[:2]
    # Arrow start point (slightly above screen center)
    start_point = (width // 2, int(height * 0.4))
    
    # Calculate arrow direction
    dx = target_center[0] - start_point[0]
    dy = target_center[1] - start_point[1]
    length = (dx**2 + dy**2)**0.5
    
    if length == 0:
        return frame
    
    # Arrow end point is target center point
    end_point = target_center
    
    # Draw arrow
    cv2.arrowedLine(frame, start_point, end_point, color, thickness, tipLength=0.3)
    return frame

def get_position_descriptor(index, total_count):
    """Return position descriptor word based on index and total count"""
    if total_count == 1:
        return ""
    elif total_count == 2:
        return "left" if index == 0 else "right"
    else:
        if index == 0:
            return "leftmost"
        elif index == total_count - 1:
            return "rightmost"
        elif total_count % 2 == 1 and index == total_count // 2:
            return "middle"
        else:
            # Determine if object is on left or right side of frame
            if index < total_count // 2:
                # On left side, from left count
                from_left = index + 1
                if from_left == 2:
                    return "second from the left"
                elif from_left == 3:
                    return "third from the left" 
                else:
                    return f"{from_left}th from the left"
            else:
                # On right side, from right count
                from_right = total_count - index
                if from_right == 2:
                    return "second from the right"
                elif from_right == 3:
                    return "third from the right"
                else:
                    return f"{from_right}th from the right"

def get_relative_position(target_box, other_boxes, other_classes):
    """Find nearest other object and describe relative position"""
    tx1, ty1, tx2, ty2 = map(int, target_box[:4])
    target_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
    
    # Find left and right nearest objects
    nearest_left = None
    nearest_right = None
    left_dist = float('inf')
    right_dist = float('inf')
    left_class = None 
    right_class = None
    
    for box, cls in zip(other_boxes, other_classes):
        x1, y1, x2, y2 = map(int, box[:4])
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Calculate horizontal distance
        dist = abs(center[0] - target_center[0])
        
        # If in target left
        if center[0] < target_center[0]:
            if dist < left_dist:
                left_dist = dist
                nearest_left = center
                left_class = cls
        # If in target right
        elif center[0] > target_center[0]:
            if dist < right_dist:
                right_dist = dist
                nearest_right = center
                right_class = cls
                
    # If no objects on either side
    if nearest_left is None and nearest_right is None:
        return ""
        
    # If only left side has object
    if nearest_right is None:
        left_class_name = [name for name, id in CLASSES.items() if id == int(left_class)][0]
        return f" at the right of the {left_class_name}"
        
    # If only right side has object
    if nearest_left is None:
        right_class_name = [name for name, id in CLASSES.items() if id == int(right_class)][0]
        return f" at the left of the {right_class_name}"
        
    # If both sides have objects
    left_class_name = [name for name, id in CLASSES.items() if id == int(left_class)][0]
    right_class_name = [name for name, id in CLASSES.items() if id == int(right_class)][0]
    
    if left_class_name == right_class_name:
        return f" between two {left_class_name}s"
    else:
        return f" between the {left_class_name} and the {right_class_name}"

def get_object_positions(class_name):
    """Get position description of specific category objects, using multi-frame buffering technique"""
    global overlay_frame  # Need to modify overlay_frame to add arrow
    
    # Get best detection result (frame with most objects)
    best_result = get_best_detection()
    
    if best_result is None:
        return "I don't see any objects currently being segmented."
        
    class_id = CLASSES.get(class_name.lower())
    if class_id is None:
        return f"I don't recognize the object type '{class_name}'."
        
    # Get all boundary boxes of this category and other categories
    target_boxes = []
    target_centers = []
    other_boxes = []
    other_classes = []
    
    height, width = frame_buffer.shape[:2]
    
    for box, cls in zip(best_result.boxes.data, best_result.boxes.cls):
        if int(cls) == class_id:
            target_boxes.append(box)
            x1, y1, x2, y2 = map(int, box[:4])
            target_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        else:
            other_boxes.append(box)
            other_classes.append(cls)
    
    if not target_boxes:
        return f"I don't see any {class_name} in the current scene."
        
    # If there is only one target object
    if len(target_boxes) == 1:
        x1, y1, x2, y2 = map(int, target_boxes[0][:4])
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        region = get_region(center[0], center[1], width, height)
        relative = get_relative_position(target_boxes[0], other_boxes, other_classes)
        
        # Add arrow to overlay
        if overlay_frame is not None:
            overlay_frame = draw_arrow(overlay_frame, (int(center[0]), int(center[1])))
        
        return f"The {class_name} is in the {region} of the screen{relative}."
        
    # If there are multiple target objects
    positions = []
    # Sort by x coordinate
    sorted_indices = sorted(range(len(target_centers)), 
                          key=lambda i: target_centers[i][0])
    
    # Add arrow to each object
    if overlay_frame is not None:
        for idx in sorted_indices:
            center = target_centers[idx]
            overlay_frame = draw_arrow(overlay_frame, (int(center[0]), int(center[1])))
    
    for i, idx in enumerate(sorted_indices):
        box = target_boxes[idx]
        center_x, center_y = target_centers[idx]
        
        # Use new position description function
        position_word = get_position_descriptor(i, len(target_boxes))
        if position_word:  # If there is position description word
            region = get_region(center_x, center_y, width, height)
            relative = get_relative_position(box, other_boxes, other_classes)
            positions.append(f"The {position_word} {class_name} is in the {region} of the screen{relative}")
    
    return " ".join(positions)

def handle_position_query(clean_prompt):
    """Handle position query"""
    prompt_lower = clean_prompt.lower()
    
    # Check if it's a position query
    location_keywords = ['where', 'location', 'position']
    if not any(keyword in prompt_lower for keyword in location_keywords):
        return False
        
    # Find mentioned categories
    for class_name in CLASSES.keys():
        if class_name in prompt_lower:
            response = get_object_positions(class_name)
            print(f'Ada: {response}')
            speak(response)
            return True
            
    return False

# Add position query handling in callback and start_listening
def callback(recognizer, audio):
    try:
        visual_context = None  # Fix undefined exception
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
            
        prompt_text = wav_to_text(prompt_audio_path)
        clean_prompt = extract_prompt(prompt_text, wake_word)
        
        if clean_prompt:
            chat_history.append(f"User: {clean_prompt}")
            print(f'USER: {clean_prompt}')
            call = function_call(clean_prompt)
            
            # Handle different function calls
            if 'position query' in call:
                if handle_position_query(clean_prompt):
                    return
            elif 'count objects' in call:
                if handle_count_query(clean_prompt):
                    return
            elif 'take screenshot' in call:
                print('Taking screenshot...')
                take_screenshot()
                visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.png')
            elif 'real-time segmentation' in call:
                segmentation_thread_instance = threading.Thread(target=segmentation_thread, kwargs={'prompt': clean_prompt})
                segmentation_thread_instance.start()
                visual_context = None
            elif 'well done' in call:
                print('Thank you...')
                stop_segmentation()
                visual_context = None
            elif 'extract clipboard' in call:
                print('Copying clipboard text...')
                paste = get_clipboard()
                clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT: {paste}'
                visual_context = None
            elif 'quit' in call.lower():
                quit_program()
            else:
                visual_context = None
            
            # Handle cases where LLM response is needed
            if ('real-time segmentation' not in call) and ('well done' not in call):
                response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
                print(f'Ada: {response}')
                speak(response)
    except Exception as e:
        print(f"Error in voice recognition: {e}")

def on_press(event):
    """Handle key press event"""
    global is_speaking, source
    try:
        # Modify space key detection method
        if event.name == 'space' or (  # First check name attribute
            hasattr(event, 'vk') and event.vk == 32) or (  # Then check virtual key code
            isinstance(event, keyboard.KeyCode) and event.char == ' '):  # Finally check character
            
            if not is_speaking:
                # Start recording
                is_speaking = True
                print("Recording started... Press space again to stop.")
                try:
                    with source as s:
                        audio = r.listen(s, timeout=None, phrase_time_limit=None)
                        # Handle recording
                        handle_audio(audio)
                except Exception as e:
                    print(f"Error recording: {e}")
                finally:
                    is_speaking = False
                    print("Recording stopped.")
            else:
                # If recording, this space press will trigger stop_listening in r.listen
                r.stop_listening()
    except Exception as e:
        print(f"Error in key press handler: {e}")

def on_release(event):
    """Handle key release event"""
    pass  # No need to handle release event

def start_listening():
    # Start video display thread
    video_thread = threading.Thread(target=display_video_thread)
    video_thread.daemon = True
    video_thread.start()
    
    # Start audio recording
    setup_audio_recording()
    
    if use_voice_interaction:
        from pynput import keyboard
        
        # Set keyboard listener
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        
        with source as s:
            print('Listening...')
            r.adjust_for_ambient_noise(s, duration=1)
            r.energy_threshold = 300
            r.dynamic_energy_threshold = True
            r.dynamic_energy_adjustment_damping = 0.15
            r.dynamic_energy_ratio = 1.5
            r.pause_threshold = 0.5
            r.operation_timeout = None
            r.phrase_threshold = 0.3
            r.non_speaking_duration = 0.3
        
        print('\nPress space to start/stop recording.\n')  # Update prompt information
        
        while True:
            time.sleep(.5)
    else:
        # Modify command line input processing part
        while True:
            user_input = input("Command: ")
            clean_prompt = extract_prompt(user_input, wake_word)
            
            if clean_prompt:
                chat_history.append(f"User: {clean_prompt}")
                print(f'User: {clean_prompt}')
                
                call = function_call(clean_prompt)
                
                # Add describe frame processing
                if 'describe frame' in call:
                    describe_frame()
                    continue
                elif 'position query' in call:
                    if handle_position_query(clean_prompt):
                        continue
                elif 'count objects' in call:
                    if handle_count_query(clean_prompt):
                        continue
                elif 'quit' in call.lower():
                    quit_program()
                    break
                
                if 'take screenshot' in call:
                    print('Taking screenshot...')
                    take_screenshot()
                    visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.png')
                elif 'real-time segmentation' in call:
                    # Start segmentation in a new thread to avoid blocking
                    segmentation_thread_instance = threading.Thread(target=segmentation_thread, kwargs={'prompt': clean_prompt})
                    segmentation_thread_instance.start()
                    visual_context = None
                elif 'well done' in call:
                    print('Thank you...')
                    stop_segmentation()
                    visual_context = None
                elif 'extract clipboard' in call:
                    print('Copying clipboard text...')
                    paste = get_clipboard()
                    clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT: {paste}'
                    visual_context = None
                elif 'visual question' in call:
                    answer_visual_question(clean_prompt)
                    continue
                else:
                    visual_context = None
                
                if ('real-time segmentation' not in call) and ('well done' not in call):
                    response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
                    print(f'Ada: {response}')
                    # Add Ada's reply to conversation history
                    chat_history.append(f"Ada: {response}")
                    speak(response)

def extract_prompt(transcribed_text, wake_word):
    """Return transcribed text directly"""
    return transcribed_text.strip() if transcribed_text else None
        
def setup_audio_recording():
    """Set audio recording"""
    global recording_stream
    
    # If recording is disabled, do not start audio recording
    if not ENABLE_RECORDING:
        return
        
    recording_stream = p_record.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    # Start audio recording thread
    audio_thread = threading.Thread(target=record_audio_thread)
    audio_thread.daemon = True
    audio_thread.start()

def record_audio_thread():
    """Thread function for continuous audio recording"""
    global audio_frames
    
    # If recording is disabled, directly return
    if not ENABLE_RECORDING:
        return
        
    while not should_exit:
        try:
            data = recording_stream.read(CHUNK, exception_on_overflow=False)
            audio_frames.append(data)
        except Exception as e:
            print(f"Error recording audio: {e}")
            continue

def save_audio_recording(base_path):
    """Save audio recording file"""
    import wave
    
    # Use same filename as video but with wav format
    audio_path = base_path.rsplit('.', 1)[0] + '_audio.wav'
    
    with wave.open(audio_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p_record.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_frames))
    
    print(f"Audio recording saved to: {audio_path}")

def quit_program():
    """Function to exit program"""
    global should_exit, stop_event, video_writer, recording_stream, p_record
    
    # Stop segmentation (if in progress)
    stop_event.set()
    
    # Say goodbye and wait for speech to complete
    speak("Goodbye! Have a great day!")
    print("Exiting program...")
    time.sleep(2)  # Wait for speech to finish
    
    # Set exit flag to close video stream
    should_exit = True
    
    # Give video stream a moment to clean up
    time.sleep(0.5)
    
    # Save video and audio
    if video_writer is not None and ENABLE_RECORDING:
        video_writer.release()
        print(f"Video recording saved to: {current_video_path}")
        
        # Stop audio recording and save
        if recording_stream is not None:
            recording_stream.stop_stream()
            recording_stream.close()
            save_audio_recording(current_video_path)
        p_record.terminate()
    
    # Exit program
    os._exit(0)
        
def handle_audio(audio):
    """Handle recorded audio"""
    try:
        visual_context = None  # Fix undefined exception
        prompt_audio_path = 'prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())
            
        prompt_text = wav_to_text(prompt_audio_path)
        clean_prompt = extract_prompt(prompt_text, wake_word)
        
        if clean_prompt:
            chat_history.append(f"User: {clean_prompt}")
            print(f'USER: {clean_prompt}')
            call = function_call(clean_prompt)
            
            # Start playing waiting sound effect
            start_waiting_sound()
            
            # Handle instance reasoning request
            if call.startswith("instance_reasoning:"):
                target_class = call.split(":", 1)[1]
                instance_reasoning(clean_prompt, target_class)
                return
            elif 'describe frame' in call:
                describe_frame()
                return
            elif 'visual question' in call:
                answer_visual_question(clean_prompt)
                return
            elif 'position query' in call:
                if handle_position_query(clean_prompt):
                    return
            elif 'count objects' in call:
                if handle_count_query(clean_prompt):
                    return
            elif 'take screenshot' in call:
                print('Taking screenshot...')
                take_screenshot()
                visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.png')
            elif 'real-time segmentation' in call:
                segmentation_thread_instance = threading.Thread(target=segmentation_thread, kwargs={'prompt': clean_prompt})
                segmentation_thread_instance.start()
                visual_context = None
            elif 'well done' in call:
                print('Thank you...')
                stop_segmentation()
                visual_context = None
            elif 'extract clipboard' in call:
                print('Copying clipboard text...')
                paste = get_clipboard()
                clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT: {paste}'
                visual_context = None
            elif 'quit' in call.lower():
                quit_program()
        else:
            visual_context = None
        
        if ('real-time segmentation' not in call) and ('well done' not in call):
            response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
            print(f'Ada: {response}')
            speak(response)
    except Exception as e:
        print(f"Error in voice recognition: {e}")
        
# Add following functions to handle "describe frame" functionality
def describe_frame():
    """Analyze and describe current video frame"""
    global frame_buffer
    
    if frame_buffer is None:
        speak("I can't see anything right now.")
        return
    
    # Save current frame
    frame_path = 'current_frame_analysis.png'
    cv2.imwrite(frame_path, frame_buffer)
    
    print("Analyzing what I can see...")
    
    try:
        # Prepare API request
        from base64 import b64encode
        
        # Read image and encode as base64
        with open(frame_path, "rb") as image_file:
            image_data = b64encode(image_file.read()).decode('utf-8')
        
        # Use Groq's visual model API
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Describe what you see in the current frame with a very concise description in one sentence."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }
        ]
        
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=messages,
            max_tokens=300
        )
        
        description = response.choices[0].message.content
        print(f"Ada: {description}")
        speak(description)
        
    except Exception as e:
        error_message = f"I encountered an error analyzing the image: {str(e)}"
        print(error_message)
        speak(error_message)
    
    # Clean up temporary files
    try:
        os.remove(frame_path)
    except:
        pass

# Add new function to answer specific questions about video frame
def answer_visual_question(question):
    """Analyze current video frame and answer user's specific question"""
    global frame_buffer
    
    if frame_buffer is None:
        speak("I can't see anything right now to answer your question.")
        return
    
    # Save current frame
    frame_path = 'current_frame_question.png'
    cv2.imwrite(frame_path, frame_buffer)
    
    print(f"Analyzing frame to answer: {question}")
    
    try:
        # Prepare API request
        from base64 import b64encode
        
        # Read image and encode as base64
        with open(frame_path, "rb") as image_file:
            image_data = b64encode(image_file.read()).decode('utf-8')
        
        # Use Groq's visual model API, including user's specific question
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": f"Look at this image and answer this question very concisely: {question}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }
        ]
        
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=messages,
            max_tokens=100
        )
        
        answer = response.choices[0].message.content
        print(f"Ada: {answer}")
        speak(answer)
        
    except Exception as e:
        error_message = f"I encountered an error analyzing the image: {str(e)}"
        print(error_message)
        speak(error_message)
    
    # Clean up temporary files
    try:
        os.remove(frame_path)
    except:
        pass
        
# Add new instance reasoning function
def instance_reasoning(question, target_class):
    """Analyze each instance of a specific category and provide description"""
    global current_results, frame_buffer
    
    if current_results is None or frame_buffer is None:
        speak("I don't have any segmentation results to analyze yet.")
        return
    
    # Get target class ID in CLASSES
    target_class_id = None
    for class_name, class_id in CLASSES.items():
        if class_name.lower() == target_class.lower():
            target_class_id = class_id
            break
    
    if target_class_id is None:
        speak(f"I don't recognize {target_class} as a known object class.")
        return
    
    # Extract all boxes of specified category
    boxes = []
    class_ids = []
    
    # Check for boxes attribute
    if hasattr(current_results, 'boxes') and current_results.boxes is not None:
        for i, cls in enumerate(current_results.boxes.cls):
            if int(cls) == target_class_id:
                box = current_results.boxes.xyxy[i].cpu().numpy()
                boxes.append(box)
                class_ids.append(int(cls))
    
    if not boxes:
        speak(f"I couldn't find any {target_class} in the current segmentation results.")
        return
    
    # Sort by x coordinate (from left to right)
    sorted_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    
    instance_descriptions = []
    position_descriptors = get_position_descriptors(len(boxes))
    
    print(f"Analyzing {len(boxes)} {target_class} instances...")
    
    # Process each instance
    for i, idx in enumerate(sorted_indices):
        box = boxes[idx]
        x1, y1, x2, y2 = map(int, box)
        
        # Ensure boundary is within image range
        height, width = frame_buffer.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Crop boundary box
        cropped_instance = frame_buffer[y1:y2, x1:x2]
        
        if cropped_instance.size == 0:
            continue
            
        # Save cropped area
        instance_path = f'instance_{i}.png'
        cv2.imwrite(instance_path, cropped_instance)
        
        # Analyze single instance
        description = analyze_instance(instance_path, target_class, position_descriptors[i])
        instance_descriptions.append(description)
        
        # Clean up temporary files
        try:
            os.remove(instance_path)
        except:
            pass
    
    # Combine all instance descriptions
    if len(instance_descriptions) == 1:
        final_description = f"The {target_class} {instance_descriptions[0]}"
    else:
        final_description = "; ".join(instance_descriptions)
    
    # Cache result to memory file
    cache_instance_reasoning(target_class, final_description)
    
    print(f"Ada: {final_description}")
    speak(final_description)
    return final_description

def analyze_instance(image_path, class_name, position):
    """Analyze single instance and return description"""
    try:
        from base64 import b64encode
        
        # Read image and encode as base64
        with open(image_path, "rb") as image_file:
            image_data = b64encode(image_file.read()).decode('utf-8')
        
        # Use Groq's visual model API to analyze instance
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": f"Analyze this cropped image of a {class_name}. Describe its notable visual features (color, condition, shape, etc.) in one brief phrase."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }
        ]
        
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=messages,
            max_tokens=50
        )
        
        features = response.choices[0].message.content.strip()
        
        # Build description
        return f"{position} is {features}"
        
    except Exception as e:
        print(f"Error analyzing instance: {e}")
        return f"{position} is visible"

def get_position_descriptors(count):
    """Generate position descriptor word based on object count"""
    if count == 1:
        return [""]
    elif count == 2:
        return ["on the left", "on the right"]
    else:
        result = []
        for i in range(count):
            if i == 0:
                result.append("on the far left")
            elif i == count - 1:
                result.append("on the far right")
            elif i == 1:
                result.append("on the left")
            elif i == count - 2:
                result.append("on the right")
            else:
                relative_position = (i / (count - 1)) * 100
                result.append(f"at position {i+1} from the left")
        return result

def cache_instance_reasoning(target_class, description):
    """Cache instance reasoning result to JSON file"""
    cache_file = "instance_memory.json"
    
    # Load existing cache or create new cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                memory = json.load(f)
        except:
            memory = {}
    else:
        memory = {}
    
    # Update cache
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    memory[timestamp] = {
        "class": target_class,
        "description": description
    }
    
    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(memory, f, indent=2)
        
start_listening()
