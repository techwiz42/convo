import { create } from 'zustand';

const useSpeechStore = create((set) => ({
  isTTSEnabled: false,
  isListening: false,
  setTTSEnabled: (enabled) => set({ isTTSEnabled: enabled }),
  setListening: (listening) => set({ isListening: listening }),
}));

class SpeechHandler {
  constructor() {
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      this.recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      this.recognition.maxAlternatives = 1;
      this.recognition.lang = 'en-US';
    }

    this.synthesis = window.speechSynthesis;
    this.isListening = false;
    this.voices = [];
    this.loadVoices();

    this.characterVoices = {
      "Hemingway": {
        preferredLang: 'en-US',
        pitch: 0.85,
        rate: 0.9,
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
      "Dickinson": {
        preferredLang: 'en-US',
        pitch: 1.2,
        rate: 0.85,
        voiceGender: 'female',
        voicePriority: ['Google US English Female', 'Microsoft Zira', 'Samantha']
      },
      "Moderator": {
        preferredLang: 'en-US',
        pitch: 1.0,
        rate: 1.0,
        voiceGender: 'neutral',
        voicePriority: ['Google US English', 'Microsoft David', 'Alex']
      }
    };

    if (this.recognition) {
      this.recognition.onresult = this.handleSpeechResult.bind(this);
      this.recognition.onend = () => {
        if (this.isListening) {
          this.recognition.start();
        }
      };
    }

    if (this.synthesis) {
      this.synthesis.addEventListener('voiceschanged', () => this.loadVoices());
    }
  }

  loadVoices() {
    this.voices = this.synthesis.getVoices();
  }

  findVoiceForCharacter(character) {
    const settings = this.characterVoices[character] || this.characterVoices["Moderator"];
    let selectedVoice = null;
    
    if (settings.voicePriority) {
      for (const priorityVoice of settings.voicePriority) {
        selectedVoice = this.voices.find(v => 
          v.name.toLowerCase().includes(priorityVoice.toLowerCase())
        );
        if (selectedVoice) break;
      }
    }

    if (!selectedVoice && this.voices.length > 0) {
      selectedVoice = this.voices[0];
    }

    return selectedVoice;
  }

  handleSpeechResult(event) {
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        const finalText = event.results[i][0].transcript;
        if (this.onSpeechCallback) {
          this.onSpeechCallback(finalText);
        }
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
    try {
      this.recognition.start();
    } catch (error) {
      if (error.name !== 'InvalidStateError') {
        throw error;
      }
    }
  }

  stopListening() {
    if (this.recognition) {
      this.isListening = false;
      this.recognition.stop();
    }
  }

  async speak(text, character = 'Moderator') {
    if (!this.synthesis) {
      alert('Speech synthesis is not supported in your browser.');
      return;
    }

    const wasListening = this.isListening;
    if (wasListening) {
      this.stopListening();
    }

    this.synthesis.cancel();

    const settings = this.characterVoices[character] || this.characterVoices["Moderator"];
    const utterance = new SpeechSynthesisUtterance(text);
    
    utterance.voice = this.findVoiceForCharacter(character);
    utterance.pitch = settings.pitch;
    utterance.rate = settings.rate;
    utterance.volume = 1.0;

    if (!utterance.voice && settings.preferredLang) {
      utterance.lang = settings.preferredLang;
    }

    return new Promise((resolve) => {
      utterance.onend = () => {
        if (wasListening) {
          setTimeout(() => {
            this.startListening(this.onSpeechCallback);
          }, 500);
        }
        resolve();
      };

      utterance.onerror = () => {
        if (wasListening) {
          this.startListening(this.onSpeechCallback);
        }
        resolve();
      };

      this.synthesis.speak(utterance);
    });
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

export { SpeechHandler, useSpeechStore };
