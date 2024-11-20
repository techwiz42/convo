// speech-handler.js
class SpeechHandler {
    constructor() {
        // Initialize speech recognition
        if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
            this.recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-US';
        }

        // Initialize speech synthesis
        this.synthesis = window.speechSynthesis;
        this.isListening = false;

        // Bind methods
        this.startListening = this.startListening.bind(this);
        this.stopListening = this.stopListening.bind(this);
        this.handleSpeechResult = this.handleSpeechResult.bind(this);
        this.speak = this.speak.bind(this);

        // Event handlers
        if (this.recognition) {
            this.recognition.onresult = this.handleSpeechResult;
            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.stopListening();
            };
            this.recognition.onend = () => {
                if (this.isListening) {
                    this.recognition.start();
                }
            };
        }
    }

    startListening(onSpeechCallback) {
        if (!this.recognition) {
            alert('Speech recognition is not supported in your browser.');
            return;
        }
        this.isListening = true;
        this.onSpeechCallback = onSpeechCallback;
        this.recognition.start();
    }

    stopListening() {
        if (this.recognition) {
            this.isListening = false;
            this.recognition.stop();
        }
    }

    handleSpeechResult(event) {
        const last = event.results.length - 1;
        const text = event.results[last][0].transcript;
        if (this.onSpeechCallback) {
            this.onSpeechCallback(text);
        }
    }

    speak(text) {
        if (!this.synthesis) {
            alert('Speech synthesis is not supported in your browser.');
            return;
        }
        
        // Cancel any ongoing speech
        this.synthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        this.synthesis.speak(utterance);
    }

    toggleSpeech(onSpeechCallback) {
        if (this.isListening) {
            this.stopListening();
        } else {
            this.startListening(onSpeechCallback);
        }
        return this.isListening;
    }
}

export default SpeechHandler;
