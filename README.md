# Swarm Chat

A FastAPI-based chat application that implements a swarm of conversational agents, each with distinct personalities inspired by famous authors and characters. The system dynamically routes conversations through different agents, creating an engaging and unpredictable chat experience.

## Features

- **Multi-Agent System**: Includes personalities such as:
  - Ernest Hemingway
  - Thomas Pynchon
  - Emily Dickinson
  - Dale Carnegie
  - A Freudian Psychoanalyst
  - A 1920s Flapper

- **Dynamic Agent Selection**: Randomly assigns conversation handlers for each session
- **Secure Session Management**: Implements token-based authentication
- **Concurrent Chat Support**: Handles multiple simultaneous chat sessions with proper locking mechanisms
- **WebSocket Integration**: Provides real-time chat functionality
- **Message History**: Maintains conversation history per user session

## Technical Architecture

### Components

1. **FastAPI Server** (`swarm_chat.py`/`swarm_chat_v1.py`)
   - Handles HTTP endpoints and WebSocket connections
   - Manages user sessions and authentication
   - Routes messages to appropriate agents

2. **Agent System** (`agents.py`)
   - Defines different agent personalities
   - Implements agent selection and routing logic
   - Manages agent state and transitions

3. **Frontend** (`static/index.html`)
   - Provides web interface for chat
   - Handles real-time updates and message display

### Key Classes

- `UserSession`: Manages individual user chat sessions
- `SwarmChatManager`: Handles session management and message routing
- `Agent`: Base class for implementing different chat personalities

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd swarm-chat
```

2. Install dependencies:
```bash
pip install fastapi uvicorn pydantic
```

3. Start the server:
```bash
python swarm_chat.py
```

The application will be available at `http://localhost:8000`

## Usage

1. Access the web interface through your browser
2. Log in with any username/password (authentication is currently simplified)
3. Start chatting with the swarm of agents

Each session will be assigned a random agent personality that will maintain its character throughout the conversation.

## Development

### Adding New Agents

To add a new agent personality:

1. Add the agent to the `authors` list in `agents.py`
2. Create a new Agent class with appropriate instructions
3. Add corresponding transfer functions
4. Update the triage agent's function list

Example:
```python
new_agent = Agent(
    name="New Author",
    instructions="Answer as New Author. Do not begin your answer with 'Ah'."
)
```

### Security Considerations

- The current implementation uses basic authentication
- Session tokens are generated using `secrets.token_urlsafe`
- Proper authentication should be implemented for production use

## Future Improvements

- Add persistent storage for chat history
- Implement proper user authentication
- Add agent-switching capabilities during conversations
- Enhance personality traits and response patterns
- Add conversation analytics
- Implement proper error handling and recovery

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
