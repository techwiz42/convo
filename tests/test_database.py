# test_database.py

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, Conversation, get_db

# Use an in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="function")
def test_db():
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_create_conversation(test_db):
    conversation = Conversation(
        user_id="test_user",
        conversation_history="Test conversation",
        last_question="Test question"
    )
    test_db.add(conversation)
    test_db.commit()

    retrieved_conversation = test_db.query(Conversation).filter_by(user_id="test_user").first()
    assert retrieved_conversation is not None
    assert retrieved_conversation.conversation_history == "Test conversation"
    assert retrieved_conversation.last_question == "Test question"

def test_update_conversation(test_db):
    conversation = Conversation(
        user_id="test_user",
        conversation_history="Initial conversation",
        last_question="Initial question"
    )
    test_db.add(conversation)
    test_db.commit()

    conversation.conversation_history += " Updated conversation"
    conversation.last_question = "Updated question"
    test_db.commit()

    retrieved_conversation = test_db.query(Conversation).filter_by(user_id="test_user").first()
    assert retrieved_conversation.conversation_history == "Initial conversation Updated conversation"
    assert retrieved_conversation.last_question == "Updated question"

def test_get_db():
    db = next(get_db())
    assert db is not None
    db.close()
