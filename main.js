// Chat Module
const Chat = {
    isVisible: false,
    messageHistory: [],

    initialize: function() {
        this.chatWidget = document.getElementById('chatWidget');
        this.messagesContainer = document.getElementById('chatMessages');
        this.userInput = document.getElementById('userMessage');
        
        // Add initial welcome message
        this.addMessage('Welcome to MindSense AI! How can I help you today? 😊', 'bot');
    },

    toggle: function() {
        this.isVisible = !this.isVisible;
        this.chatWidget.style.display = this.isVisible ? 'flex' : 'none';
        if (this.isVisible) {
            this.userInput.focus();
        }
    },

    addMessage: function(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        messageDiv.textContent = text;
        this.messagesContainer.appendChild(messageDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        this.messageHistory.push({ text, sender });
    },

    sendMessage: async function() {
        const message = this.userInput.value.trim();
        if (!message) return;

        this.addMessage(message, 'user');
        this.userInput.value = '';
        this.showTypingIndicator();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();
            this.hideTypingIndicator();

            if (data.success) {
                this.addMessage(data.response, 'bot');
            } else {
                this.addMessage('I apologize, but I encountered an error. Please try again.', 'bot');
            }
        } catch (error) {
            console.error('Chat error:', error);
            this.hideTypingIndicator();
            this.addMessage('I apologize, but I\'m having trouble connecting. Please try again.', 'bot');
        }
    },

    showTypingIndicator: function() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-typing';
        typingDiv.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>`;
        typingDiv.id = 'typingIndicator';
        this.messagesContainer.appendChild(typingDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    },

    hideTypingIndicator: function() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    },

    handleKeyPress: function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }
};

// Text Analysis Module
const TextAnalysis = {
    emotionColors: {
        'joy': '#FFD700',
        'sadness': '#4169E1',
        'anger': '#FF4500',
        'fear': '#800080',
        'surprise': '#00CED1',
        'neutral': '#808080',
        'love': '#FF69B4'
    },

    emotionIcons: {
        'joy': 'fa-smile-beam',
        'sadness': 'fa-sad-tear',
        'anger': 'fa-angry',
        'fear': 'fa-ghost',
        'surprise': 'fa-surprise',
        'neutral': 'fa-meh',
        'love': 'fa-heart'
    },

    analyzeText: async function() {
        const textInput = document.querySelector('.analysis-input').value.trim();
        const resultsDiv = document.getElementById('text-results');
        
        // Clear previous results
        resultsDiv.innerHTML = '';
        
        if (!textInput) {
            this.showError('Please enter some text to analyze');
            return;
        }
        
        // Show loading state
        resultsDiv.innerHTML = `
            <div class="loading">
                <div class="loading-spinner"></div>
                <p>Analyzing your text...</p>
            </div>
        `;
        resultsDiv.style.display = 'block';
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: textInput })
            });
            
            const data = await response.json();
            
            if (data.success && data.predictions && data.predictions.length > 0) {
                const emotion = data.predictions[0];
                
                // Create emotion display div
                const emotionDiv = document.createElement('div');
                emotionDiv.className = 'emotion-result';
                emotionDiv.style.backgroundColor = `${emotion.color}22`;
                emotionDiv.style.border = `2px solid ${emotion.color}`;
                
                emotionDiv.innerHTML = `
                    <div class="emotion-header">
                        <i class="fas ${emotion.icon} fa-2x" style="color: ${emotion.color}"></i>
                        <h3>${this.capitalizeFirstLetter(emotion.emotion)}</h3>
                        <p>Confidence: ${emotion.confidence}</p>
                    </div>
                    <div class="recommendations-preview">
                        <div class="preview-item">
                            <i class="fas fa-music" style="color: ${emotion.color}"></i>
                            <span>${emotion.music[0]}</span>
                        </div>
                        <div class="preview-item">
                            <i class="fas fa-lungs" style="color: ${emotion.color}"></i>
                            <span>Breathing exercises available</span>
                        </div>
                    </div>
                    <a href="/recommendations" class="view-recommendations-btn" style="background: ${emotion.color}">
                        <i class="fas fa-arrow-right"></i> View All Recommendations
                    </a>
                `;
                
                resultsDiv.innerHTML = '';
                resultsDiv.appendChild(emotionDiv);
                resultsDiv.style.display = 'block';
            } else {
                this.showError(data.error || 'Failed to analyze emotions');
            }
        } catch (error) {
            console.error('Error:', error);
            this.showError('An error occurred while analyzing the text');
        }
    },
    
    showError: function(message) {
        const resultsDiv = document.getElementById('text-results');
        resultsDiv.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                ${message}
            </div>
        `;
        resultsDiv.style.display = 'block';
    },

    capitalizeFirstLetter: function(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
};

// Voice Analysis Module
const VoiceAnalysis = {
    recognition: null,
    isListening: false,

    initialize: function() {
        if ('webkitSpeechRecognition' in window) {
            this.recognition = new webkitSpeechRecognition();
            this.setupRecognition();
        } else {
            console.error('Speech recognition is not supported in this browser');
        }
    },

    setupRecognition: function() {
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';

        this.recognition.onstart = () => {
            this.isListening = true;
            this.updateStatus('Listening...', 'listening');
        };

        this.recognition.onend = () => {
            this.isListening = false;
            this.updateStatus('Click to start listening', 'stopped');
        };

        this.recognition.onresult = (event) => {
            let interimTranscript = '';
            let finalTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                    this.analyzeEmotion(finalTranscript);
                } else {
                    interimTranscript += transcript;
                }
            }

            document.getElementById('liveTranscription').textContent = finalTranscript;
            document.getElementById('interimTranscription').textContent = interimTranscript;
        };

        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.updateStatus('Error: ' + event.error, 'error');
        };
    },

    startListening: function() {
        if (!this.isListening && this.recognition) {
            this.recognition.start();
        }
    },

    stopListening: function() {
        if (this.isListening && this.recognition) {
            this.recognition.stop();
        }
    },

    updateStatus: function(message, status) {
        const statusContainer = document.querySelector('.status-container');
        if (statusContainer) {
            statusContainer.className = `status-container ${status}`;
            document.getElementById('statusText').textContent = message;
        }
    },

    analyzeEmotion: async function(text) {
        if (!text.trim()) return;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();
            if (data.success) {
                TextAnalysis.displayResults(data.predictions);
            }
        } catch (error) {
            console.error('Error analyzing voice emotion:', error);
        }
    }
};

// Initialize everything when the page loads
document.addEventListener('DOMContentLoaded', function() {
    Chat.initialize();
    VoiceAnalysis.initialize();

    // Global function for chat toggle
    window.toggleChat = function() {
        Chat.toggle();
    };

    // Global function for sending messages
    window.sendMessage = function() {
        Chat.sendMessage();
    };

    // Global function for handling key press
    window.handleKeyPress = function(event) {
        Chat.handleKeyPress(event);
    };
}); 