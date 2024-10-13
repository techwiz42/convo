# test_user_knowledge_base.py

import pytest
from unittest.mock import MagicMock, patch
from user_knowledge_base import UserKnowledgeBase

@pytest.fixture
def mock_db_session():
    with patch('user_knowledge_base.get_db') as mock_get_db:
        mock_session = MagicMock()
        mock_get_db.return_value = iter([mock_session])
        yield mock_session

def test_add_knowledge(mock_db_session):
    kb = UserKnowledgeBase("test_user")
    kb.add_knowledge("topic", "content")

    mock_db_session.query.assert_called_once()
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()

def test_get_relevant_knowledge(mock_db_session):
    kb = UserKnowledgeBase("test_user")
    mock_db_session.query().filter_by().first.return_value = MagicMock(conversation_history="Test conversation")

    result = kb.get_relevant_knowledge("test query")

    assert isinstance(result, list)
    assert len(result) > 0

def test_get_relevant_knowledge_empty(mock_db_session):
    kb = UserKnowledgeBase("test_user")
    mock_db_session.query().filter_by().first.return_value = None

    result = kb.get_relevant_knowledge("test query")

    assert isinstance(result, list)
    assert len(result) == 0

def test_update_tfidf(mock_db_session):
    kb = UserKnowledgeBase("test_user")
    mock_db_session.query().filter_by().first.return_value = MagicMock(conversation_history="Test conversation")

    kb._update_tfidf()

    assert kb.tfidf_matrix is not None

def test_ensure_fitted():
    kb = UserKnowledgeBase("test_user")
    kb._ensure_fitted()

    assert kb.tfidf_matrix is not None
