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

- **Dynamic Agent Selection**: Randomly assigns conversation handlers after each user response.
- **Secure Session Management**: Implements token-based authentication
- **Concurrent Chat Support**: Handles multiple simultaneous chat sessions with proper locking mechanisms
- **WebSocket Integration**: Provides real-time chat functionality
- **Message History**: Maintains conversation history per user session

## Technical Architecture

### Components

1. **FastAPI Server** (`swarm_chat.py`)
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
cd swarm
```

2. Install dependencies:
```bash
pip install fastapi uvicorn pydantic
or
setup.py -r requirements.txt
```

3. Start the server:
```bash
python swarm_chat.py
```

The application will be available at `http://localhost:8000`

## Usage

1. Access the web interface through your browser
2. Log in with any username - password not required (authentication is currently simplified)
3. Start chatting with the swarm of agents

Each session will be assigned a random agent personality. The agent will be changed randomly during the department.

## Testing
A comprehensive test suite exists. To run tests:
~~~
# Run all tests
pytest -v tests/test_swarm_chat.py

# Run only performance tests
pytest -v -m performance tests/test_swarm_chat.py

# Run only security tests
pytest -v -m security tests/test_swarm_chat.py
~~~
To run all tests in a class or a single test method in a class
~~~
# Run all tests in TestMessageHandling class
pytest tests/test_swarm_chat.py::TestMessageHandling -v

# Run all tests with "MessageHandling" in the name
pytest -v -k "MessageHandling"

# Run just the test_basic_message test in TestMessageHandling
pytest tests/test_swarm_chat.py::TestMessageHandling::test_basic_message -v
~~~
Omit the -v for less verbose output.

Before checking in: 
* run pylint to verify pep-8 code compliance
* run mypy for syntax and static type check

## Deployment
The code runs as swarmchat.service in a DigitalOcean droplet. 
It runs as a service in the NGINX web server.
To restart the service   
~~~
>sudo systemctl restart swarmchat.service
~~~
Check service status like so
~~~
>sudo systemctl status swarmchat.service
~~~
And if the status is not 'running', get error output like so:
~~~
>journalctl -u swarmchat -n 25
~~~
So far, the project is deployed manually as the procedure is not
overly involved. Future development work would include creating
a CI/CD pipeline.

## Development
lots to do.

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

None

## Contributing

Fork the repo and go nuts.
