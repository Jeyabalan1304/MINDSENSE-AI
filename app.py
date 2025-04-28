from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory, make_response
import torch
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from functools import lru_cache
import time
import os
import librosa
import requests
import json
import traceback
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = 'mindsense_ai_secret_key'

# Set HuggingFace token as environment variable
os.environ["HUGGINGFACE_TOKEN"] = "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"
HF_TOKEN = os.environ["HUGGINGFACE_TOKEN"]

# Create temp directories if they don't exist
if not os.path.exists('temp_audio'):
    os.makedirs('temp_audio')
if not os.path.exists('model_cache'):
    os.makedirs('model_cache')

# GPT API Configuration
GPT_API_KEY = "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"  # Updated Groq API Key
GPT_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Initialize models and configurations
class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_classifier = None
        self.whisper_processor = None
        self.whisper_model = None
        self.voice_emotion_model = None
        self.voice_emotion_processor = None
        
    def initialize_models(self):
        print("Initializing models...")
        try:
            # Initialize emotion classifier
            print("Loading emotion classifier...")
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
                top_k=None,
                device=self.device
            )
            print("Emotion classifier loaded successfully!")

            # Initialize Whisper model with proper error handling
            print("Loading Whisper model...")
            try:
                self.whisper_processor = WhisperProcessor.from_pretrained(
                    "openai/whisper-base",
                    cache_dir="./model_cache"
                )
                self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                    "openai/whisper-base",
                    cache_dir="./model_cache"
                ).to(self.device)
                print(f"Whisper model loaded successfully and moved to {self.device}!")
            except Exception as e:
                print(f"Error loading Whisper model: {str(e)}")
                traceback.print_exc()
                return False

            # Initialize wav2vec2 model for voice emotion recognition
            print("Loading wav2vec2 voice emotion model...")
            try:
                from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
                
                self.voice_emotion_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                    cache_dir="./model_cache"
                )
                self.voice_emotion_model = AutoModelForAudioClassification.from_pretrained(
                    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                    cache_dir="./model_cache"
                ).to(self.device)
                print(f"Voice emotion model loaded successfully and moved to {self.device}!")
            except Exception as e:
                print(f"Error loading voice emotion model: {str(e)}")
                traceback.print_exc()
                return False

            # Verify all models are loaded
            if (self.emotion_classifier and self.whisper_processor and 
                self.whisper_model and self.voice_emotion_processor and 
                self.voice_emotion_model):
                print("All models initialized successfully!")
                return True
            else:
                print("Some models failed to initialize")
                return False

        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            traceback.print_exc()
            return False

# Create model manager instance
model_manager = ModelManager()

# Enhanced emotion mapping with more natural responses
EMOTION_MAPPING = {
    'joy': {
        'music': [
            '🎵 Upbeat Pop Music - <a href="https://www.youtube.com/watch?v=k1mG8QN62Oo" target="_blank">Listen Now</a>',
            '🎸 Happy Rock Songs - <a href="https://www.youtube.com/watch?v=IaU8iFKLX40" target="_blank">Listen Now</a>',
            '🎺 Jazz Fusion - <a href="https://www.youtube.com/watch?v=6yq5J0_i6HM" target="_blank">Listen Now</a>',
            '🌟 Dance & Electronic - <a href="https://www.youtube.com/watch?v=lnELeMNdOBc" target="_blank">Listen Now</a>'
        ],
        'breathing': 'Hey, your energy is amazing! Let\'s keep that positive vibe going. Try this simple breathing exercise: breathe in for 4 seconds, hold that happiness in for 4 seconds, and let it flow out for 4 seconds. It\'s like riding the wave of joy! 🌊',
        'color': '#FFD700',
        'icon': 'fa-smile-beam'
    },
    'sadness': {
        'music': [
            '🎹 Soothing Piano Melodies',
            '🎻 Gentle Classical Pieces',
            '🌊 Calming Nature Sounds',
            '🎸 Acoustic Comfort Songs'
        ],
        'breathing': 'Hey, I hear you. It\'s okay to feel this way. Let\'s try something together - take a deep breath with me. Breathe in slowly for 5 seconds (like you\'re smelling a flower), hold for 2 seconds, then let it all out for 6 seconds (like you\'re blowing out a candle). It\'s like giving yourself a gentle hug from the inside. 💙',
        'color': '#4169E1',
        'icon': 'fa-sad-tear'
    },
    'anger': {
        'music': [
            '🎵 Peaceful Melodies',
            '🌊 Ocean Wave Sounds',
            '🎸 Mellow Rock',
            '🎹 Calming Jazz'
        ],
        'breathing': 'I get it - sometimes things can be really frustrating. Let\'s try this together: breathe in through your mouth for 4 seconds (like cooling down hot soup), hold that breath for 4 seconds, then slowly release through your nose for 6 seconds. It\'s like letting the steam out safely. 🫂',
        'color': '#FF4500',
        'icon': 'fa-angry'
    },
    'fear': {
        'music': [
            '🎵 Gentle Meditation Tunes',
            '🌊 Soft Ocean Waves',
            '🎹 Comforting Piano',
            '🍃 Peaceful Nature Sounds'
        ],
        'breathing': 'Hey, I\'m right here with you. You\'re safe. Let\'s ground ourselves together with this breathing technique: breathe in for 4 seconds (like filling yourself with strength), hold for 7 seconds (feeling stable and secure), then exhale for 8 seconds (letting go of those worries). You\'ve got this! 🤗',
        'color': '#800080',
        'icon': 'fa-ghost'
    },
    'surprise': {
        'music': [
            '🎵 Upbeat Pop Music - <a href="https://www.youtube.com/watch?v=k1mG8QN62Oo" target="_blank">Listen Now</a>',
            '🎸 Happy Rock Songs - <a href="https://www.youtube.com/watch?v=IaU8iFKLX40" target="_blank">Listen Now</a>',
            '🎺 Jazz Fusion - <a href="https://www.youtube.com/watch?v=6yq5J0_i6HM" target="_blank">Listen Now</a>',
            '🌟 Dance & Electronic - <a href="https://www.youtube.com/watch?v=lnELeMNdOBc" target="_blank">Listen Now</a>'
        ],
        'breathing': 'Wow, what a moment! Let\'s channel this energy with some balanced breathing: breathe in for 4 seconds, hold briefly, then exhale for 4 seconds. It\'s like riding this wave of surprise with grace! ✨',
        'color': '#00CED1',
        'icon': 'fa-surprise'
    },
    'neutral': {
        'music': [
            '🎵 Upbeat Pop Music - <a href="https://www.youtube.com/watch?v=k1mG8QN62Oo" target="_blank">Listen Now</a>',
            '🎸 Happy Rock Songs - <a href="https://www.youtube.com/watch?v=IaU8iFKLX40" target="_blank">Listen Now</a>',
            '🎺 Jazz Fusion - <a href="https://www.youtube.com/watch?v=6yq5J0_i6HM" target="_blank">Listen Now</a>',
            '🌟 Dance & Electronic - <a href="https://www.youtube.com/watch?v=lnELeMNdOBc" target="_blank">Listen Now</a>'
        ],
        'breathing': 'Let\'s enhance your state of mind with this simple square breathing: breathe in for 4 seconds, hold for 4 seconds, exhale for 4 seconds, and hold again for 4 seconds. It\'s like finding your perfect rhythm. 🌟',
        'color': '#808080',
        'icon': 'fa-meh'
    },
    'love': {
        'music': [
            '🎵 Heart-Warming Ballads',
            '🎹 Romantic Jazz',
            '🎻 Beautiful Classical',
            '🎸 Sweet Acoustic Songs'
        ],
        'breathing': 'What a beautiful feeling! Let\'s embrace it fully: breathe in love for 4 seconds (like embracing warmth), hold that wonderful feeling for 4 seconds, then share it with the world as you exhale for 4 seconds. It\'s like spreading joy everywhere! 💖',
        'color': '#FF69B4',
        'icon': 'fa-heart'
    }
}

@lru_cache(maxsize=1000)
def get_text_emotion_prediction(text):
    try:
        if model_manager.emotion_classifier is None:
            raise Exception("Emotion classifier not properly initialized")

        # Get emotion predictions
        emotions = model_manager.emotion_classifier(text)[0]
        print(f"Raw emotion predictions: {emotions}")  # Debug log
        
        # Convert to our format with recommendations
        results = []
        
        # Map the model's emotion labels to our emotion categories
        emotion_label_mapping = {
            'joy': 'joy',
            'sadness': 'sadness',
            'anger': 'anger',
            'fear': 'fear',
            'surprise': 'surprise',
            'neutral': 'neutral',
            'disgust': 'anger',  # Map disgust to anger
            'happy': 'joy',      # Map happy to joy
            'sad': 'sadness'     # Map sad to sadness
        }
        
        for emotion in emotions:
            emotion_name = emotion_label_mapping.get(emotion['label'].lower(), 'neutral')
            confidence = f"{emotion['score']*100:.1f}%"
            
            # Get recommendations from mapping
            emotion_data = EMOTION_MAPPING.get(emotion_name, EMOTION_MAPPING['neutral'])
            
            results.append({
                'emotion': emotion_name,
                'confidence': confidence,
                'music': emotion_data['music'],
                'breathing': emotion_data['breathing'],
                'color': emotion_data['color'],
                'icon': emotion_data['icon']
            })
        
        # Sort by confidence and get only the top emotion
        results.sort(key=lambda x: float(x['confidence'].strip('%')), reverse=True)
        top_result = results[0]  # Get only the highest confidence emotion
        print(f"Top emotion with recommendations: {top_result}")  # Debug log
        return [top_result]
        
    except Exception as e:
        print(f"Error in emotion prediction: {str(e)}")
        traceback.print_exc()
        return None

# Improved chatbot response generation
def generate_chatbot_response(user_message, emotion_predictions):
    try:
        # Get the primary emotion and its data
        primary_emotion = emotion_predictions[0]
        emotion_name = primary_emotion['emotion']
        
        # Create more natural, context-aware system prompts
        system_prompts = {
            'joy': "You are a warm, friendly person who genuinely shares in others' happiness. Be enthusiastic and encouraging, like a close friend celebrating good news. Use casual, upbeat language and appropriate emojis to keep the positive energy flowing.",
            'sadness': "You are a compassionate friend providing comfort during difficult times. Be gentle, understanding, and validating of their feelings. Share words of comfort and hope, like a caring friend who's there to listen and support. Use calming language and appropriate emojis to show you care.",
            'anger': "You are a calm, understanding friend helping someone process their frustration. Acknowledge their feelings without judgment, like a trusted friend who helps put things in perspective. Use steady, grounding language and appropriate emojis to help diffuse tension.",
            'fear': "You are a reassuring friend providing comfort during anxious moments. Be steady and supportive, like a trusted companion who helps others feel safe. Use calming language and appropriate emojis to create a sense of security.",
            'surprise': "You are an engaged friend sharing in moments of amazement. Be genuinely interested and responsive, like a friend who loves discovering new things together. Use expressive language and appropriate emojis to share in the excitement.",
            'neutral': "You are a friendly companion having a casual conversation. Be natural and engaging, like a friend chatting over coffee. Use conversational language and occasional emojis to keep things warm and relatable.",
            'love': "You are a warmhearted friend sharing in moments of joy and connection. Be genuinely happy for others, like a close friend celebrating life's beautiful moments. Use warm language and appropriate emojis to enhance the positive feelings."
        }
        
        # Prepare the API request with updated format for Groq
        headers = {
            "Authorization": f"Bearer {GPT_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompts.get(emotion_name, system_prompts['neutral']) + "\nRemember to: 1) Use natural, conversational language 2) Show genuine empathy 3) Share personal-feeling insights 4) Offer gentle support 5) Keep responses concise and friendly"
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "temperature": 0.8,
            "max_tokens": 150
        }
        
        try:
            response = requests.post(GPT_API_URL, headers=headers, json=data, timeout=15)
            print(f"API Response Status: {response.status_code}")  # Debug log
            
            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    chatbot_response = response_data['choices'][0]['message']['content'].strip()
                    
                    # Add natural emoji based on emotion
                    emoji_map = {
                        'joy': ' 😊',
                        'sadness': ' 💙',
                        'anger': ' 🫂',
                        'fear': ' 🤗',
                        'surprise': ' 😮',
                        'love': ' 💖',
                        'neutral': ' 👋'
                    }
                    
                    # Only add emoji if it feels natural and isn't already present
                    if emoji_map.get(emotion_name) and emoji_map[emotion_name] not in chatbot_response:
                        chatbot_response += emoji_map[emotion_name]
                    
                    return {
                        'response': chatbot_response,
                        'success': True
                    }
                else:
                    raise Exception("No response choices available")
            else:
                error_message = f"API Error: Status code {response.status_code}"
                print(error_message)
                if response.text:
                    print(f"Error details: {response.text}")
                raise Exception(error_message)
            
        except requests.exceptions.RequestException as e:
            print(f"API request error: {str(e)}")
            raise e
            
    except Exception as e:
        print(f"Error in chatbot response generation: {str(e)}")
        return {
            'response': "I'm here to listen and chat whenever you're ready. Let's try again in a moment. 🤗",
            'success': True
        }

@app.route('/')
def landing():
    return render_template('about.html')

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

def get_user_by_username(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    return user

def create_user(username, password, email):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        hashed_password = generate_password_hash(password, method='sha256')
        c.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                 (username, hashed_password, email))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        if not username or not password or not email:
            return render_template('signup.html', error="All fields are required")
        
        # Check if username already exists
        if get_user_by_username(username):
            return render_template('signup.html', error="Username already exists")
        
        # Create new user
        if create_user(username, password, email):
            return redirect(url_for('login', message="Account created successfully! Please login."))
        else:
            return render_template('signup.html', error="Error creating account")
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember')
        
        user = get_user_by_username(username)
        
        if user and check_password_hash(user[2], password):
            session['logged_in'] = True
            session['username'] = username
            
            response = make_response(redirect(url_for('dashboard')))
            
            if remember:
                # Set a longer session expiry (30 days)
                session.permanent = True
                app.permanent_session_lifetime = timedelta(days=30)
                
                # Set a remember_me cookie
                response.set_cookie('remember_me', username, max_age=30*24*60*60)
            
            return response
        else:
            return render_template('login.html', error="Invalid credentials")
    
    # Check for remember_me cookie
    remember_cookie = request.cookies.get('remember_me')
    if remember_cookie:
        user = get_user_by_username(remember_cookie)
        if user:
            session['logged_in'] = True
            session['username'] = remember_cookie
            return redirect(url_for('dashboard'))
    
    return render_template('login.html', message=request.args.get('message'))

@app.route('/logout')
def logout():
    session.clear()
    response = make_response(redirect(url_for('landing')))
    response.delete_cookie('remember_me')
    return response

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'})
    
    try:
        text = request.json.get('text', '').strip()
        print(f"Analyzing text: {text}")  # Debug log
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'})
        
        predictions = get_text_emotion_prediction(text)
        if predictions is None:
            return jsonify({'success': False, 'error': 'Failed to analyze emotions'})
        
        # Store the emotion prediction in session
        session['last_emotion'] = predictions[0]
        
        # Format the response
        response = {
            'success': True,
            'predictions': predictions
        }
        print(f"Sending response: {response}")  # Debug log
        return jsonify(response)
        
    except Exception as e:
        error_message = f"Prediction error: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_message})

@app.route('/chat', methods=['POST'])
def chat():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'})
    
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Please enter a message to chat'
            })
        
        print(f"Received message: {user_message}")  # Debug log
        
        # Get emotion prediction with enhanced error handling
        try:
            emotion_predictions = get_text_emotion_prediction(user_message)
            if emotion_predictions is None:
                raise Exception("Failed to analyze emotions")
            print(f"Detected emotion: {emotion_predictions[0]['emotion']}")  # Debug log
        except Exception as e:
            print(f"Emotion prediction error: {str(e)}")
            # Use neutral emotion as fallback
            emotion_predictions = [{
                'emotion': 'neutral',
                'confidence': '100%'
            }]
        
        # Generate chatbot response
        chat_response = generate_chatbot_response(user_message, emotion_predictions)
        print(f"Chat response: {chat_response}")  # Debug log
        
        if not chat_response.get('success'):
            return jsonify({
                'success': False,
                'error': 'Failed to generate response'
            })
        
        return jsonify({
            'success': True,
            'response': chat_response['response']
        })
        
    except Exception as e:
        error_message = f"Chat endpoint error: {str(e)}"
        print(error_message)
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing your message. Please try again.'
        })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# Add new route for recommendations page
@app.route('/recommendations')
def recommendations():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # Get the last predicted emotion from session
    emotion_data = session.get('last_emotion', None)
    if not emotion_data:
        return redirect(url_for('dashboard'))
    
    # Only show professional support for negative emotions
    positive_emotions = ['joy', 'happy', 'surprise']
    if emotion_data['emotion'] not in positive_emotions:
        return redirect(url_for('professional_support'))
    
    return render_template('recommendations.html', emotion_data=emotion_data)

# Add new route for professional support page
@app.route('/professional-support')
def professional_support():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    emotion_data = session.get('last_emotion', None)
    if not emotion_data:
        return redirect(url_for('dashboard'))
    
    return render_template('professional_support.html', emotion_data=emotion_data)

# Add new route for specialists map page
@app.route('/specialists')
def specialists():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('specialists.html')

# Add new route for chatbot page
@app.route('/chatbot')
def chatbot():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('chatbot.html')

# Add new route for detailed recommendations page
@app.route('/detailed-recommendations')
def detailed_recommendations():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    emotion_data = session.get('last_emotion', None)
    if not emotion_data:
        return redirect(url_for('dashboard'))
    
    return render_template('detailed_recommendations.html', emotion_data=emotion_data)

@app.route('/process_voice', methods=['POST'])
def process_voice():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'})
    
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'})
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})
        
        # Get file extension
        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        
        # Save the audio file temporarily with original extension
        filename = secure_filename(f"voice_{int(time.time())}{file_ext}")
        filepath = os.path.join('temp_audio', filename)
        
        print(f"Saving audio file to: {filepath}")
        audio_file.save(filepath)
        
        try:
            # Convert audio to WAV format if needed
            print("Loading audio file with librosa...")
            y, sr = librosa.load(filepath, sr=16000, mono=True)
            
            # Save as WAV temporarily
            wav_filepath = os.path.join('temp_audio', f"temp_{int(time.time())}.wav")
            import soundfile as sf
            sf.write(wav_filepath, y, sr, format='WAV')
            
            # Process the WAV file
            result = process_audio_for_emotion(wav_filepath)
            
            # Clean up temporary files
            try:
                os.remove(filepath)
                os.remove(wav_filepath)
            except Exception as e:
                print(f"Error cleaning up temporary files: {str(e)}")
            
            if result['success']:
                # Store the emotion prediction in session
                session['last_emotion'] = {
                    'emotion': result['emotion'],
                    'confidence': result['confidence'],
                    'music': EMOTION_MAPPING[result['emotion']]['music'],
                    'breathing': EMOTION_MAPPING[result['emotion']]['breathing'],
                    'color': EMOTION_MAPPING[result['emotion']]['color'],
                    'icon': EMOTION_MAPPING[result['emotion']]['icon']
                }
                
                return jsonify({
                    'success': True,
                    'emotion': result['emotion'],
                    'confidence': result['confidence'],
                    'transcribed_text': result['transcribed_text'],
                    'all_emotions': result['all_emotions'],
                    'recommendations': {
                        'music': EMOTION_MAPPING[result['emotion']]['music'],
                        'breathing': EMOTION_MAPPING[result['emotion']]['breathing'],
                        'color': EMOTION_MAPPING[result['emotion']]['color'],
                        'icon': EMOTION_MAPPING[result['emotion']]['icon']
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Failed to analyze emotion')
                })
                
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Error processing audio: {str(e)}'
            })
            
    except Exception as e:
        print(f"Error in process_voice route: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing the audio file'
        })

def process_audio_for_emotion(audio_file_path):
    try:
        if not model_manager.whisper_processor or not model_manager.whisper_model or not model_manager.emotion_classifier:
            raise Exception("Required models not properly initialized")
            
        print(f"Processing audio file: {audio_file_path}")
        
        # Load and preprocess audio file with error handling
        try:
            print("Loading audio file...")
            waveform, sample_rate = librosa.load(audio_file_path, sr=16000, mono=True)
            print(f"Audio loaded - Length: {len(waveform)}, Sample Rate: {sample_rate}")
            
            # Ensure minimum audio length and handle silence
            if len(waveform) < sample_rate * 0.5:  # Less than 0.5 seconds
                raise Exception("Audio too short - please speak for at least 0.5 seconds")
            
            if np.abs(waveform).max() < 0.01:  # Check if audio is too quiet
                raise Exception("No speech detected - please speak louder")
            
            # Normalize audio
            waveform = librosa.util.normalize(waveform)
            
        except Exception as e:
            print(f"Error loading audio: {str(e)}")
            raise Exception(f"Error processing audio file: {str(e)}")
        
        try:
            print("Converting audio to text...")
            # Convert audio to text using Whisper with proper error handling
            input_features = model_manager.whisper_processor(
                waveform, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(model_manager.device)
            
            # Generate token ids with proper settings
            predicted_ids = model_manager.whisper_model.generate(
                input_features,
                max_length=448,
                min_length=1,
                num_beams=5,
                length_penalty=1.0,
                temperature=0.7
            )
            
            # Decode the token ids to text
            transcribed_text = model_manager.whisper_processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            print(f"Transcribed text: {transcribed_text}")
            
            if not transcribed_text:
                raise Exception("No speech detected - please try speaking again")
            
        except Exception as e:
            print(f"Error in speech recognition: {str(e)}")
            raise Exception(f"Error in speech recognition: {str(e)}")
        
        try:
            print("Analyzing emotions...")
            # Get emotion predictions from the transcribed text
            emotions = model_manager.emotion_classifier(transcribed_text)[0]
            print(f"Raw emotion predictions: {emotions}")
            
            # Map the model's emotion labels to our emotion categories
            emotion_label_mapping = {
                'joy': 'joy',
                'sadness': 'sadness',
                'anger': 'anger',
                'fear': 'fear',
                'surprise': 'surprise',
                'neutral': 'neutral',
                'disgust': 'anger',  # Map disgust to anger
                'happy': 'joy',      # Map happy to joy
                'sad': 'sadness'     # Map sad to sadness
            }
            
            # Process all emotions with confidence scores
            all_emotions = {}
            for emotion in emotions:
                emotion_name = emotion_label_mapping.get(emotion['label'].lower(), 'neutral')
                confidence = emotion['score'] * 100
                all_emotions[emotion_name] = confidence
            
            # Get the highest confidence emotion
            top_emotion = max(emotions, key=lambda x: x['score'])
            emotion_name = emotion_label_mapping.get(top_emotion['label'].lower(), 'neutral')
            confidence = top_emotion['score'] * 100
            
            result = {
                'success': True,
                'emotion': emotion_name,
                'confidence': f"{confidence:.1f}%",
                'transcribed_text': transcribed_text,
                'all_emotions': all_emotions
            }
            
            print(f"Final result: {result}")
            return result
            
        except Exception as e:
            print(f"Error in emotion analysis: {str(e)}")
            raise Exception(f"Error analyzing emotions: {str(e)}")
            
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

@app.route('/analyze_voice_emotion', methods=['POST'])
def analyze_voice_emotion():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'})
    
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'})
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'})
        
        # Validate file type
        if not audio_file.filename.lower().endswith('.wav'):
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload WAV files only.'})
        
        # Save the audio file temporarily
        filename = secure_filename(f"voice_{int(time.time())}.wav")
        filepath = os.path.join('temp_audio', filename)
        
        print(f"Saving audio file to: {filepath}")
        audio_file.save(filepath)
        
        try:
            # Load and process the audio file
            print("Loading audio file...")
            waveform, sample_rate = librosa.load(filepath, sr=16000, mono=True)
            
            # Process audio with wav2vec2 model
            inputs = model_manager.voice_emotion_processor(
                waveform, 
                sampling_rate=sample_rate, 
                return_tensors="pt",
                padding=True
            ).to(model_manager.device)
            
            # Get emotion predictions
            with torch.no_grad():
                outputs = model_manager.voice_emotion_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get emotion labels from the model's config
            emotion_labels = model_manager.voice_emotion_model.config.id2label
            
            # Convert predictions to emotion probabilities
            emotions = {}
            for i, prob in enumerate(predictions[0]):
                emotion = emotion_labels[i]
                confidence = float(prob) * 100
                emotions[emotion] = confidence
            
            # Get the dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            
            result = {
                'success': True,
                'dominant_emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1],
                'emotions': emotions
            }
            
            # Clean up temporary file
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Error cleaning up temporary file: {str(e)}")
            
            return jsonify(result)
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Error processing audio: {str(e)}'
            })
            
    except Exception as e:
        print(f"Error in analyze_voice_emotion route: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An error occurred while analyzing the voice emotion'
        })

if __name__ == '__main__':
    print("Starting server...")
    print("Initial model load will take a few seconds...")
    
    # Initialize models
    if model_manager.initialize_models():
        print("All models initialized successfully!")
        app.run(debug=True)
    else:
        print("Error initializing models. Please check the logs above.")
        exit(1)  # Exit with error code if models fail to initialize 