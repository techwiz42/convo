import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader } from 'lucide-react';

const ConversationalChatbot = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    // Add user message
    const userMessage = {
      text: inputText,
      sender: 'user',
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await fetch('http://queequeg.local:8000/convert', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: inputText,
          context: messages.map(m => m.text).join('\n'),
          temperature: 0.7,
          top_p: 0.9,
        }),
      });

      const data = await response.json();
      
      // Add bot message
      const botMessage = {
        text: data.converted_text,
        sender: 'bot',
        timestamp: new Date().toISOString(),
        isQuestion: data.input_type === 'statement', // bot response is question if input was statement
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      // Add error message
      const errorMessage = {
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
        timestamp: new Date().toISOString(),
        isError: true,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto p-4 bg-gray-50">
      <div className="bg-white rounded-lg shadow-md p-4 mb-4">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">Conversational Chatbot</h1>
        <p className="text-gray-600">Ask questions or make statements - I'll respond appropriately!</p>
      </div>

      <div className="flex-1 overflow-y-auto mb-4 bg-white rounded-lg shadow-md p-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`mb-4 ${
              message.sender === 'user' ? 'ml-auto' : 'mr-auto'
            }`}
          >
            <div
              className={`max-w-sm rounded-lg p-4 ${
                message.sender === 'user'
                  ? 'bg-blue-500 text-white ml-auto'
                  : message.isError
                  ? 'bg-red-100 text-red-700'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              <p className="mb-1">{message.text}</p>
              <div className="text-xs opacity-75">
                {new Date(message.timestamp).toLocaleTimeString()}
                {message.sender === 'bot' && !message.isError && (
                  <span className="ml-2">
                    {message.isQuestion ? '(Question)' : '(Statement)'}
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Type a question or statement..."
          className="flex-1 p-4 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading}
          className="p-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <Loader className="w-6 h-6 animate-spin" />
          ) : (
            <Send className="w-6 h-6" />
          )}
        </button>
      </form>
    </div>
  );
};

export default ConversationalChatbot;
