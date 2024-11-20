// speech-handler.js
class SpeechHandler {
    constructor() {
        // Initialize speech recognition
        if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
            this.recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            
            // Recognition settings
            this.recognition.continuous = true;        // Keep listening after results
            this.recognition.interimResults = true;    // Get results while speaking
            this.recognition.maxAlternatives = 1;      
            this.recognition.lang = 'en-US';
            
            // Only tracking interim results now
            this.interimTranscript = '';
        }

        // Initialize speech synthesis
        this.synthesis = window.speechSynthesis;
        this.isListening = false;
        this.voices = [];
        this.loadVoices();

        // Character voice settings with browser-compatible voices
        this.characterVoices = {
            "Hemmingway": {
                preferredLang: 'en-US',
                pitch: 0.85,    // Deep voice
                rate: 0.9,      // Measured pace
                voiceGender: 'male',
                voicePriority: ['Google US English', 'Microsoft David', 'Alex']
            },
            "Pynchon": {
                preferredLang: 'en-US',
                pitch: 1.1,
                rate: 1.15,
                voiceGender: 'male',
                voicePriority: ['Google US English', 'Microsoft Mark', 'Alex']
            },
            "Emily Dickenson": {
                preferredLang: 'en-US',
                pitch: 1.2,
                rate: 0.85,
                voiceGender: 'female',
                voicePriority: ['Google US English Female', 'Microsoft Zira', 'Samantha']
            },
            "Dale Carnegie": {
                preferredLang: 'en-US',
                pitch: 1.05,
                rate: 1.1,
                voiceGender: 'male',
                voicePriority: ['Google US English', 'Microsoft David', 'Alex']
            },
            "H. L. Mencken": {
                preferredLang: 'en-US',
                pitch: 0.95,
                rate: 1.05,
                voiceGender: 'male',
                voicePriority: ['Microsoft Mark', 'Google US English', 'Alex']
            },
            "A Freudian Psychoanalyst": {
                preferredLang: 'de-DE',
                pitch: 0.9,
                rate: 0.9,
                voiceGender: 'male',
                voicePriority: ['Google Deutsch', 'Microsoft Stefan', 'Anna']
            },
            "A flapper from the 1920s": {
                preferredLang: 'en-US',
                pitch: 1.3,
                rate: 1.2,
                voiceGender: 'female',
                voicePriority: ['Google US English Female', 'Microsoft Zira', 'Samantha']
            },
            "Bullwinkle J. Moose": {
                preferredLang: 'en-US',
                pitch: 0.7,
                rate: 0.95,
                voiceGender: 'male',
                voicePriority: ['Google US English', 'Microsoft David', 'Alex']
            },
            "Yogi Berra": {
                preferredLang: 'en-US',
                pitch: 1.0,
                rate: 0.95,
                voiceGender: 'male',
                voicePriority: ['Google US English', 'Microsoft Mark', 'Alex']
            },
            "Harbhajan Singh Khalsa": {
                preferredLang: 'hi-IN',
                pitch: 0.95,
                rate: 0.9,
                voiceGender: 'male',
                voicePriority: ['Google हिन्दी', 'Microsoft Hemant', 'Microsoft Heera']
            },
            "Moderator": {
                preferredLang: 'en-US',
                pitch: 1.0,
                rate: 1.0,
                voiceGender: 'neutral',
                voicePriority: ['Google US English', 'Microsoft David', 'Alex']
            }
        };

        // Bind methods
        this.startListening = this.startListening.bind(this);
        this.stopListening = this.stopListening.bind(this);
        this.handleSpeechResult = this.handleSpeechResult.bind(this);
        this.speak = this.speak.bind(this);

        // Event handlers for recognition
        if (this.recognition) {
            this.recognition.onresult = this.handleSpeechResult;
            
            this.recognition.onend = () => {
                console.log('Speech recognition ended');
                if (this.isListening) {
                    console.log('Restarting speech recognition');
                    this.recognition.start();
                }
            };

            this.recognition.onerror = (event) => {
                console.log('Speech recognition error:', event.error);
                if (event.error === 'no-speech') {
                    // Don't stop listening on no-speech error, just restart
                    if (this.isListening) {
                        console.log('Restarting after no-speech error');
                        this.recognition.start();
                    }
                } else {
                    console.error('Speech recognition error:', event.error);
                    this.stopListening();
                }
            };

            // Audio monitoring events
            this.recognition.onaudiostart = () => console.log('Audio capturing started');
            this.recognition.onaudioend = () => console.log('Audio capturing ended');
            this.recognition.onsoundstart = () => console.log('Sound detected');
            this.recognition.onsoundend = () => console.log('Sound ended');
        }

        // Load voices when they're ready
        if (this.synthesis) {
            this.synthesis.addEventListener('voiceschanged', () => this.loadVoices());
        }
    }

    loadVoices() {
        this.voices = this.synthesis.getVoices();
        console.log('Available voices:', this.voices.map(v => ({
            name: v.name,
            lang: v.lang,
            default: v.default,
            localService: v.localService,
            voiceURI: v.voiceURI
        })));
    }

    findVoiceForCharacter(character) {
        const settings = this.characterVoices[character] || this.characterVoices["Moderator"];
        let selectedVoice = null;
        
        // First try exact matches from priority list
        if (settings.voicePriority) {
            for (const priorityVoice of settings.voicePriority) {
                selectedVoice = this.voices.find(v => 
                    v.name.toLowerCase().includes(priorityVoice.toLowerCase())
                );
                if (selectedVoice) break;
            }
        }

        // If no priority match, try to match by language and gender
        if (!selectedVoice) {
            const langVoices = this.voices.filter(v => 
                v.lang.startsWith(settings.preferredLang)
            );
            
            if (langVoices.length > 0) {
                // Try to match gender if specified
                if (settings.voiceGender) {
                    selectedVoice = langVoices.find(v => 
                        v.name.toLowerCase().includes(settings.voiceGender)
                    );
                }
                // If no gender match or gender not specified, take first matching language
                if (!selectedVoice) {
                    selectedVoice = langVoices[0];
                }
            }
        }

        // Final fallback to any available voice
        if (!selectedVoice && this.voices.length > 0) {
            selectedVoice = this.voices[0];
        }

        console.log(`Selected voice for ${character}:`, 
            selectedVoice ? {
                name: selectedVoice.name,
                lang: selectedVoice.lang,
                gender: settings.voiceGender
            } : 'No voice found'
        );

        return selectedVoice;
    }

    handleSpeechResult(event) {
        // Process each result immediately instead of accumulating
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                const finalText = event.results[i][0].transcript;
                console.log('Final transcript:', finalText);
                if (this.onSpeechCallback) {
                    this.onSpeechCallback(finalText);
                }
            } else {
                this.interimTranscript = event.results[i][0].transcript;
                console.log('Interim transcript:', this.interimTranscript);
            }
        }
    }

    startListening(onSpeechCallback) {
        if (!this.recognition) {
            alert('Speech recognition is not supported in your browser.');
            return;
        }
        this.isListening = true;
        this.onSpeechCallback = onSpeechCallback;
        this.interimTranscript = '';
        try {
            this.recognition.start();
            console.log('Speech recognition started');
        } catch (error) {
            if (error.name === 'InvalidStateError') {
                console.log('Recognition already started, continuing...');
            } else {
                throw error;
            }
        }
    }

    stopListening() {
        if (this.recognition) {
            this.isListening = false;
            this.recognition.stop();
            console.log('Speech recognition stopped');
            this.interimTranscript = '';
        }
    }

    speak(text, character = 'Moderator') {
        if (!this.synthesis) {
            alert('Speech synthesis is not supported in your browser.');
            return;
        }
        
        console.log(`Attempting to speak as character: ${character}`);
        
        // Cancel any ongoing speech
        this.synthesis.cancel();

        const settings = this.characterVoices[character] || this.characterVoices["Moderator"];
        const utterance = new SpeechSynthesisUtterance(text);
        
        // Apply character-specific settings
        utterance.voice = this.findVoiceForCharacter(character);
        utterance.pitch = settings.pitch;
        utterance.rate = settings.rate;
        utterance.volume = 1.0;

        // If we have a language but no matching voice, at least set the accent
        if (!utterance.voice && settings.preferredLang) {
            utterance.lang = settings.preferredLang;
        }

        // Debug logging
        console.log('Speaking with settings:', {
            character,
            voice: utterance.voice?.name,
            lang: utterance.voice?.lang || utterance.lang,
            pitch: utterance.pitch,
            rate: utterance.rate
        });

        // Add event handlers for debugging
        utterance.onstart = () => console.log('Started speaking');
        utterance.onend = () => console.log('Finished speaking');
        utterance.onerror = (e) => console.error('Speech error:', e);

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
