document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked tab and corresponding content
            tab.classList.add('active');
            document.getElementById(`${tab.dataset.tab}-tab`).classList.add('active');
        });
    });

    // Text Analysis Form
    const emotionForm = document.getElementById('emotionForm');
    const resultsDiv = document.getElementById('results');

    if (emotionForm) {
        emotionForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            resultsDiv.innerHTML = `
                <div class="loading">
                    <i class="fas fa-spinner fa-spin"></i>
                    Analyzing text...
                </div>`;
            
            const textInput = document.getElementById('textInput').value;
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: textInput })
                });
                
                const data = await response.json();
                displayResults(data);
            } catch (error) {
                console.error('Error:', error);
                displayError('Could not complete the analysis. Please try again.');
            }
        });
    }

    // Audio Upload Form
    const audioUploadForm = document.getElementById('audioUploadForm');
    const audioPreview = document.getElementById('audioPreview');

    if (audioUploadForm) {
        audioUploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const audioFile = document.getElementById('audioFile').files[0];
            if (!audioFile) {
                displayError('Please select an audio file.');
                return;
            }

            resultsDiv.innerHTML = `
                <div class="loading">
                    <i class="fas fa-spinner fa-spin"></i>
                    Analyzing audio...
                </div>`;

            const formData = new FormData();
            formData.append('audio', audioFile);

            try {
                const response = await fetch('/analyze-voice', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                displayResults(data);
            } catch (error) {
                console.error('Error:', error);
                displayError('Could not analyze the audio. Please try again.');
            }
        });

        // Audio file preview
        document.getElementById('audioFile').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                audioPreview.innerHTML = `
                    <audio controls>
                        <source src="${URL.createObjectURL(file)}" type="${file.type}">
                        Your browser does not support the audio element.
                    </audio>`;
            }
        });
    }

    function displayResults(data) {
        if (data.success) {
            let html = `
                <h3>
                    <i class="fas fa-chart-bar"></i>
                    Voice Emotion Analysis Results
                </h3>`;
            
            data.predictions.forEach(prediction => {
                const emoji = getEmotionEmoji(prediction.emotion);
                html += `
                    <div class="emotion-result">
                        <div class="emotion-info">
                            <span class="emotion-emoji">${emoji}</span>
                            <strong>${prediction.emotion}</strong>
                        </div>
                        <div class="confidence">
                            ${prediction.confidence}
                        </div>
                    </div>`;
            });
            
            resultsDiv.innerHTML = html;
        } else {
            resultsDiv.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-circle"></i>
                    Error: ${data.error}
                </div>`;
        }
    }

    function displayError(message) {
        resultsDiv.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                Error: ${message}
            </div>`;
    }

    function getEmotionEmoji(emotion) {
        const emojis = {
            'anger': '😠',
            'joy': '😊',
            'sadness': '😢',
            'neutral': '😐',
            'fear': '😨',
            'disgust': '🤢',
            'surprise': '😮'
        };
        return emojis[emotion.toLowerCase()] || '🤔';
    }
});
