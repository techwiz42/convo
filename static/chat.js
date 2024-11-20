// chat.js
import SpeechHandler from './speech-handler.js';

class ChatInterface {
    constructor() {
        this.speechHandler = new SpeechHandler();
        this.isTTSEnabled = false;
        
        // DOM elements
        this.toggleSpeechButton = document.getElementById('toggle-speech');
        this.toggleTTSButton = document.getElementById('toggle-tts');
        this.messageInput = document.getElementById('message-input');
        
        // Initialize speech controls
        this.initializeSpeechControls();
    }

    initializeSpeechControls() {
        // Voice input toggle
        this.toggleSpeechButton.addEventListener('click', () => {
            const isListening = this.speechHandler.toggleSpeech((text) => {
                this.messageInput.value = text;
                this.sendMessage(text);
            });
            
            this.toggleSpeechButton.classList.toggle('active', isListening);
            this.toggleSpeechButton.querySelector('.button-text').textContent = 
                isListening ? 'Stop Voice Input' : 'Start Voice Input';
        });

        // Text-to-speech toggle
        this.toggleTTSButton.addEventListener('click', () => {
            this.isTTSEnabled = !this.isTTSEnabled;
            this.toggleTTSButton.classList.toggle('active', this.isTTSEnabled);
            this.toggleTTSButton.querySelector('.button-text').textContent = 
                this.isTTSEnabled ? 'Disable Text-to-Speech' : 'Enable Text-to-Speech';
        });
    }

    async sendMessage(content) {
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ content }),
            });

            const data = await response.json();
            if (data.response && this.isTTSEnabled) {
                this.speechHandler.speak(data.response);
            }

            return data;
        } catch (error) {
            console.error('Error sending message:', error);
            throw error;
        }
    }
}

// Initialize the chat interface
const chatInterface = new ChatInterface();
