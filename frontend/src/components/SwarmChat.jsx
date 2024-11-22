import React, { useState, useRef, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./ui/dialog";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Card } from "./ui/card";
import { MessageSquare, Send, LogOut, Github, Mail, Mic, Volume2, VolumeX } from 'lucide-react';
import { twMerge } from "tailwind-merge";
import { clsx } from "clsx";
import { SpeechHandler, useSpeechStore } from '../lib/SpeechHandler';
const cn = (...inputs) => twMerge(clsx(inputs));
const API_BASE_URL = process.env.REACT_APP_API_URL || 'https://dev.swarmchat.me/api';

const SwarmChat = () => {
  const [isIntroOpen, setIsIntroOpen] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [username, setUsername] = useState('');
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const messagesEndRef = useRef(null);
  const [token, setToken] = useState(null);
  const speechHandlerRef = useRef(null);
  const { isTTSEnabled, isListening, setTTSEnabled, setListening } = useSpeechStore();

  useEffect(() => {
    speechHandlerRef.current = new SpeechHandler();
    
    if (window.speechSynthesis) {
      const handleSpeechStart = () => setIsSpeaking(true);
      const handleSpeechEnd = () => setIsSpeaking(false);
      
      window.speechSynthesis.addEventListener('start', handleSpeechStart);
      window.speechSynthesis.addEventListener('end', handleSpeechEnd);
      
      return () => {
        window.speechSynthesis.removeEventListener('start', handleSpeechStart);
        window.speechSynthesis.removeEventListener('end', handleSpeechEnd);
      };
    }
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleLogin = async (e) => {
    e?.preventDefault();
    if (!username.trim()) return;

    try {
      setIsLoading(true);
      setError(null);
      
      const credentials = btoa(`${username}:dummy`);
      
      const response = await fetch(`${API_BASE_URL}/login`, {
        method: 'POST',
        headers: {
          'Authorization': `Basic ${credentials}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`Login failed: ${errorData}`);
      }

      const data = await response.json();
      setToken(data.token);
      setIsConnected(true);
      await fetchHistory(data.token);
    } catch (err) {
      setError(`Connection error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchHistory = async (currentToken) => {
    try {
      const response = await fetch(`${API_BASE_URL}/history`, {
        headers: {
          'Authorization': `Bearer ${currentToken}`
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch history: ${response.statusText}`);
      }

      const data = await response.json();
      setMessages(data.messages);
    } catch (error) {
      setError('Failed to load chat history. Please try refreshing.');
    }
  };

  const handleSendMessage = async (e, autoSend = false) => {
    e?.preventDefault();
    const currentMessage = inputMessage;
    if (!currentMessage.trim() || !token || isLoading) return;

    try {
      setIsLoading(true);
      setError(null);
      
      setInputMessage('');

      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ content: currentMessage })
      });

      if (!response.ok) {
        throw new Error(`Failed to send message: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.response) {
        setMessages(prev => [...prev, 
          { role: 'user', content: currentMessage },
          { role: 'assistant', content: data.response }
        ]);
        
        if (isTTSEnabled && speechHandlerRef.current) {
          try {
            await speechHandlerRef.current.speak(data.response);
          } catch (err) {
            console.error('TTS Error:', err);
          }
        }
      }
    } catch (err) {
      setError(`Failed to send message: ${err.message}`);
      setInputMessage(currentMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    setIsConnected(false);
    setToken(null);
    setMessages([]);
    setUsername('');
    setError(null);
    if (speechHandlerRef.current && isListening) {
      speechHandlerRef.current.stopListening();
      setListening(false);
    }
  };

  const toggleSpeechRecognition = () => {
    if (speechHandlerRef.current) {
      const newListeningState = speechHandlerRef.current.toggleSpeech((text, autoSend) => {
        setInputMessage(text);
        if (autoSend) {
          handleSendMessage({ preventDefault: () => {} }, true);
        }
      });
      setListening(newListeningState);
    }
  };

  const toggleTTS = () => {
    if (window.speechSynthesis) {
      if (isTTSEnabled) {
        window.speechSynthesis.cancel();
      }
      setTTSEnabled(!isTTSEnabled);
    }
  };

const ActivityIndicator = () => {
    if (isListening) {
      return (
        <div className="flex justify-start">
          <div className="bg-red-50 text-red-900 max-w-[80%] p-3 rounded-lg">
            <div className="flex items-center gap-2">
              <Mic className="w-4 h-4 animate-pulse text-red-500" />
              <div className="flex gap-2">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-red-500 rounded-full animate-bounce [animation-delay:-.3s]" />
                <div className="w-2 h-2 bg-red-500 rounded-full animate-bounce [animation-delay:-.5s]" />
              </div>
            </div>
          </div>
        </div>
      );
    }
    if (isLoading) {
      return (
        <div className="flex justify-start">
          <div className="bg-gray-100 text-gray-900 max-w-[80%] p-3 rounded-lg">
            <div className="flex gap-2">
              <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" />
              <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:-.3s]" />
              <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:-.5s]" />
            </div>
          </div>
        </div>
      );
    }
    if (isSpeaking) {
      return (
        <div className="flex justify-start">
          <div className="bg-blue-100 text-blue-900 max-w-[80%] p-3 rounded-lg flex items-center gap-2">
            <Volume2 className="w-4 h-4 animate-pulse" />
            <span>Speaking...</span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <Dialog open={isIntroOpen} onOpenChange={setIsIntroOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Welcome to Swarm Chat!</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <p>This is a unique chat experience where you'll interact with a variety of AI personalities, including:</p>
            <ul className="list-disc pl-6 space-y-2">
              <li>Ernest Hemingway - Known for direct, terse prose</li>
              <li>Thomas Pynchon - Complex, postmodern style</li>
              <li>Emily Dickinson - Poetic and contemplative</li>
              <li>Dale Carnegie - Motivational and positive</li>
              <li>H. L. Mencken - A somewhat caustic journalist</li>
              <li>A Freudian Psychoanalyst - Deep psychological insights</li>
              <li>...and many more</li>
            </ul>
          </div>
        </DialogContent>
      </Dialog>

      <div className="bg-white shadow-sm p-4 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className={cn(
            "w-3 h-3 rounded-full",
            isConnected ? "bg-green-500" : "bg-red-500",
            isLoading && "animate-pulse"
          )} />
          <span>{isConnected ? 'Connected' : 'Not Connected'}</span>
        </div>
        {isConnected && (
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={toggleSpeechRecognition}
              className={cn(isListening && "bg-blue-100")}
            >
              <Mic className="w-4 h-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={toggleTTS}
              className={cn(isTTSEnabled && "bg-blue-100")}
            >
              {isTTSEnabled ? (
                <Volume2 className={cn("w-4 h-4", isSpeaking && "animate-pulse")} />
              ) : (
                <VolumeX className="w-4 h-4" />
              )}
            </Button>
            <Button variant="ghost" onClick={handleLogout} disabled={isLoading}>
              <LogOut className="w-4 h-4 mr-2" />
              Exit Session
            </Button>
          </div>
        )}
      </div>

      <div className="flex-1 container mx-auto max-w-4xl p-4">
        {!isConnected ? (
          <Card className="p-6 space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold">Login to Swarm Chat</h2>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="icon"
                  type="button"
                  onClick={toggleSpeechRecognition}
                  className={cn(isListening && "bg-blue-100")}
                  title="Speech-to-text"
                >
                  <Mic className="w-4 h-4" />
                </Button>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={toggleTTS}
                  className={cn(isTTSEnabled && "bg-blue-100")}
                  title="Text-to-speech"
                >
                  {isTTSEnabled ? (
                    <Volume2 className={cn("w-4 h-4", isSpeaking && "animate-pulse")} />
                  ) : (
                    <VolumeX className="w-4 h-4" />
                  )}
                </Button>
              </div>
            </div>
            <form onSubmit={handleLogin} className="space-y-4">
              <div className="space-y-2">
                <label className="block text-sm font-medium">Username</label>
                <Input
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Enter any name to start"
                  disabled={isLoading}
                />
                <p className="text-sm text-gray-500">
                  No account required - just enter any name to start chatting!
                  {isListening && <span className="ml-2 text-blue-500">Listening...</span>}
                </p>
              </div>
              {error && (
                <div className="p-3 text-sm text-red-600 bg-red-50 rounded-md">
                  {error}
                </div>
              )}
              <Button type="submit" disabled={isLoading}>
                <MessageSquare className="w-4 h-4 mr-2" />
                {isLoading ? 'Connecting...' : 'Start Chatting'}
              </Button>
            </form>
          </Card>
        ) : (
          <Card className="h-[calc(100vh-12rem)]">
            <div className="h-full flex flex-col">
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={cn(
                        "max-w-[80%] p-3 rounded-lg",
                        msg.role === 'user'
                          ? "bg-blue-500 text-white"
                          : "bg-gray-100 text-gray-900"
                      )}
                    >
                      {msg.content}
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
                <ActivityIndicator />
              </div>
              <div className="p-4 border-t">
                <form onSubmit={handleSendMessage} className="flex gap-2">
                  <Input
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    placeholder="Type your message..."
                    className="flex-1"
                    disabled={isLoading}
                  />
                  <Button type="submit" disabled={isLoading}>
                    <Send className="w-4 h-4" />
                  </Button>
                </form>
                {error && (
                  <p className="text-sm text-red-500 mt-2">{error}</p>
                )}
              </div>
            </div>
          </Card>
        )}
      </div>

      <footer className="bg-white shadow-sm p-4 mt-auto">
        <div className="container mx-auto max-w-4xl flex justify-center gap-4 text-sm text-gray-600">
          <a
            href="https://github.com/techwiz42/convo"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center hover:text-gray-900"
          >
            <Github className="w-4 h-4 mr-1" />
            View on GitHub
          </a>
          <span>|</span>
          <a
            href="mailto:peter.sisk2@gmail.com"
            className="flex items-center hover:text-gray-900"
          >
            <Mail className="w-4 h-4 mr-1" />
            Contact Developer
          </a>
        </div>
      </footer>
    </div>
  );
};

export default SwarmChat;
