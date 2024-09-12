import asyncio
from uuid import uuid4
from typing import Dict, Any

class DialogueManager:
    def __init__(self, model_path: str):
        # Similar initialization as before
        self.sessions: Dict[str, Dict[str, Any]] = {}

    async def create_session(self) -> str:
        session_id = str(uuid4())
        self.sessions[session_id] = {"context": ""}
        return session_id

    async def generate_question(self, session_id: str, answer: str = None) -> str:
        context = self.sessions[session_id]["context"]
        # Generate question using the model
        question = "Generated question"  # Placeholder
        self.sessions[session_id]["context"] += f" {question} {answer}"
        return question

    async def handle_dialogue(self, session_id: str, answer: str = None) -> str:
        if session_id not in self.sessions:
            return "Invalid session"
        return await self.generate_question(session_id, answer)

async def main():
    manager = DialogueManager("model_path")
    session1 = await manager.create_session()
    session2 = await manager.create_session()

    # Simulate concurrent dialogues
    q1 = await manager.handle_dialogue(session1)
    q2 = await manager.handle_dialogue(session2)

    print(f"Session 1: {q1}")
    print(f"Session 2: {q2}")

if __name__ == "__main__":
    asyncio.run(main())
