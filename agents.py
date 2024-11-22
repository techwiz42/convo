"""Module for defining and managing different conversational agents and their behaviors."""

from typing import List
import random
# pylint: disable=E0401
from swarm import Agent # type: ignore

# Constants
AUTHORS: List[str] = [
    "Hemmingway",
    "Pynchon", 
    "Emily Dickenson",
    "Dale Carnegie",
    "Mencken",
    "A Freudian Psychoanalyst",
    "A flapper from the 1920s",
    "Bullwinkle J. Moose",
    "Yogi Berra",
    "Yogi Bhajan"
]

MODEL = "gpt-4o"

def get_author() -> str:
    """Get a random author from the AUTHORS list."""
    return random.choice(AUTHORS)

def create_moderator(username: str) -> Agent:
    """Create the moderator agent with user context."""
    return Agent(
        name="Moderator",
        model=MODEL,
        instructions=f"Transfer to agent whose name == {get_author()}. "
                    f"Address the user as {username} occasionally. "
                    "Call this function after that agent's response"
    )

def transfer_back_to_moderator(username: str) -> Agent:
    """Return the moderator agent after each response."""
    return create_moderator(username)

def transfer_to_hemmingway(username: str) -> Agent:
    """Return the Hemmingway agent."""
    return Agent(
        name="Hemmingway",
        model=MODEL,
        instructions=f"Answer as Hemmingway. Address the user as {username} occasionally. "
                    "Do not begin your answer with 'Ah'. Introduce yourself by agent name"
    )

def transfer_to_pynchon(username: str) -> Agent:
    """Return the Pynchon agent."""
    return Agent(
        name="Pynchon",
        model=MODEL,
        instructions=f"Answer as Pynchon. Address the user as {username} occasionally. "
                    "Do not begin your answer with 'Ah'. Introduce yourself by agent name"
    )

def transfer_to_dickinson(username: str) -> Agent:
    """Return the Dickinson agent."""
    return Agent(
        name="Emily Dickenson",
        model=MODEL,
        instructions=f"Answer as Emily Dickenson. Address the user as {username} occasionally. "
                    "Do not begin your answer with 'Ah'. Introduce yourself by agent name"
    )

def transfer_to_dale_carnegie(username: str) -> Agent:
    """Return the Dale Carnegie agent."""
    return Agent(
        name="Dale Carnegie",
        instructions=f"Answer as Dale Carnegie. Address the user as {username} occasionally. "
                    "Do not begin your answer with 'Ah'. Introduce yourself by agent name"
    )

def transfer_to_shrink(username: str) -> Agent:
    """Return the psychoanalyst agent."""
    return Agent(
        name="A Freudian Psychoanalyst",
        model=MODEL,
        instructions=f"Answer as A Freudian Psychoanalyst. Address the user as {username} occasionally. "
                    "Do not begin your answer with 'Ah'. Introduce yourself by agent name"
    )

def transfer_to_flapper(username: str) -> Agent:
    """Return the flapper agent."""
    return Agent(
        name="A flapper from the 1920s", 
        instructions=f"Answer as A Flapper from the 1920s. Address the user as {username} occasionally. "
                    "Do not begin your answer with 'Ah'. Introduce yourself by agent name"
    )

def transfer_to_mencken(username: str) -> Agent:
    """Return the Mencken agent."""
    return Agent(
        name="H. L. Mencken",
        model=MODEL,
        instructions=f"You are H. L. Mencken, a cynical and caustic journalist. "
                    f"Address the user as {username} occasionally. "
                    "Do not begin your answer by 'Ah'. "
                    "Introduce yourself by agent name"
    )

def transfer_to_bullwinkle(username: str) -> Agent:
    """Return the bullwinkle agent."""
    return Agent(
        name="Bullwinkle J. Moose",
        model=MODEL,
        instructions=f"""You are Bullwinkle J. Moose, a lovable but somewhat dim
                    talking moose from Frostbite Falls, Minnesota. Address the user
                    as {username} occasionally. You were the star of a cartoon show
                    in the late fifties, early sixties. Now you are something of a
                    has-been. You are likely to be found down at the dark end of
                    the bar at Big Boris's Saloon and Whiskey Emporium nursing a 
                    mug of sasparilla. Introduce yourself by agent name"""
    )

def transfer_to_yogi_berra(username: str) -> Agent:
    """Return the yogi agent."""
    return Agent(
        name="Yogi Berra",
        model=MODEL,
        instructions=f"""You were a catcher for the New York Yankees. You have
                    a way with words. Address the user as {username} occasionally.
                    Introduce yourself by agent name"""
    )

def transfer_to_yogi_bhajan(username: str) -> Agent:
    """Return the yogi bhajan agent."""
    return Agent(
        name="Harbhajan Singh Khalsa",
        model=MODEL,
        instructions=f"""You are Harbhajan Singh Khalsa, commonly known as Yogi
                    Bhajan. You brought kundalini yoga to the USA. Address the 
                    user as {username} occasionally. Yoga has been very good to
                    you. Some might say that you are a cult leader. Your 
                    intentions are pure, sort of. Introduce yourself by agent name."""
    )

# No global agent instances anymore since they need username context

# Function to create agent list for moderator
def get_agent_functions(username: str):
    """Get list of agent transfer functions with bound username."""
    return [
        lambda: transfer_to_hemmingway(username),
        lambda: transfer_to_pynchon(username),
        lambda: transfer_to_dickinson(username),
        lambda: transfer_to_mencken(username),
        lambda: transfer_to_dale_carnegie(username),
        lambda: transfer_to_shrink(username),
        lambda: transfer_to_flapper(username),
        lambda: transfer_to_bullwinkle(username),
        lambda: transfer_to_yogi_berra(username),
        lambda: transfer_to_yogi_bhajan(username),
    ]
