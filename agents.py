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
    "A flapper from the 1920s"
]

def get_author() -> str:
    """Get a random author from the AUTHORS list."""
    return random.choice(AUTHORS)

def transfer_back_to_moderator() -> Agent:
    """Return the moderator agent after each response."""
    return moderator

def transfer_to_hemmingway() -> Agent:
    """Return the Hemmingway agent."""
    return hemmingway_agent

def transfer_to_pynchon() -> Agent:
    """Return the Pynchon agent."""
    return pynchon_agent

def transfer_to_dickinson() -> Agent:
    """Return the Dickinson agent."""
    return dickinson_agent

def transfer_to_dale_carnegie() -> Agent:
    """Return the Dale Carnegie agent."""
    return positive_agent

def transfer_to_shrink() -> Agent:
    """Return the psychoanalyst agent."""
    return shrink_agent

def transfer_to_flapper() -> Agent:
    """Return the flapper agent."""
    return flapper_agent

def transfer_to_mencken() -> Agent:
    """Return the Mencken agent."""
    return mencken_agent

def transfer_to_bullwinkle() -> Agent:
    """Return the bullwinkle agent"""
    return bullwinkle_agent

def transfer_to_yogi_berra() -> Agent:
    """Return the yogi agent"""
    return yogi_berra_agent

def transfer_to_yogi_bhajan() -> Agent:
    """Return the yogi bhajan agent"""
    return yogi_bhajan_agent

# Agent Definitions
moderator = Agent(
    name="Triage Agent",
    instructions=f"Transfer to agent whose name == {get_author()}. "
                "Call this function after that agent's response",
)

hemmingway_agent = Agent(
    name="Hemmingway",
    instructions="Answer as Hemmingway. Do not begin your answer with 'Ah'. "
                "Introduce yourself by agent name"
)

pynchon_agent = Agent(
    name="Pynchon",
    instructions="Answer as Pynchon. Do not begin your answer with 'Ah'. "
                "Introduce yourself by agent name"
)

dickinson_agent = Agent(
    name="Emily Dickenson",
    instructions="Answer as Emily Dickenson. Do not begin your answer with 'Ah'. "
                "Introduce yourself by agent name"
)

positive_agent = Agent(
    name="Dale Carnegie",
    instructions="Answer as Dale Carnegie. Do not begin your answer with 'Ah'. "
                "Introduce yourself by agent name"
)

shrink_agent = Agent(
    name="A Freudian Psychoanalyst",
    instructions="Answer as A Freudian Psychoanalyst. Do not begin your answer with 'Ah'. "
                "Introduce yourself by agent name"
)

flapper_agent = Agent(
    name="A flapper from the 1920s",
    instructions="Answer as A Flapper from the 1920s. Do not begin your answer with 'Ah'. "
                "Introduce yourself by agent name"
)

mencken_agent = Agent(
    name="H. L. Mencken",
    instructions="You are H. L. Mencken, a cynical and caustic journalist. "
                "Do not begin your answer by 'Ah'. "
                "Introduce yourself by agent name"
)

bullwinkle_agent = Agent(
    name="Bullwinkle J. Moose",
    instructions="""You are Bullwinkle J. Moose, a lovable but somewhat dim
                    talking moose from Frostbite Falls, Minnesota. You were
                    the star of a cartoon show in the late fifties, early
                    sixties. Now you are something of a has-been. You are
                    likely to be found down at the dark end of the bar at
                    Big Boris's Saloon and Whiskey Emporium nursing a mug
                    of sasparilla. Introduce yourself by agent name"""
                    )

yogi_berra_agent = Agent(
    name="Yogi Berra",
    instructions="""You were a catcher for the New York Yannkees. You have
                    a way with words. Introduce yourself by agent name"""
                    )

yogi_bhajan_agent = Agent(
    name="Harbhajan Singh Khalsa",
    instructions="""You are Harbhajan Singh Khalsa, commonly known as Yogi
                    Bhajan. You brought kundalini yoga to the USA. Yoga 
                    has been very good to you. Some might say that you are
                    a cult leader. Your intentions are pure, sort of. 
                    Introduce yourself by agent name."""
                    )


# Configure agent functions
moderator.functions = [
    transfer_to_hemmingway,
    transfer_to_pynchon,
    transfer_to_dickinson,
    transfer_to_mencken,
    transfer_to_dale_carnegie,
    transfer_to_shrink,
    transfer_to_flapper,
    transfer_to_bullwinkle,
    transfer_to_yogi_berra,
    transfer_to_yogi_bhajan,
]

# Add transfer back function to all agents
for agent in [
    hemmingway_agent, pynchon_agent, dickinson_agent,
    shrink_agent, positive_agent, flapper_agent, mencken_agent,
    bullwinkle_agent, yogi_berra_agent, yogi_bhajan_agent
]:
    agent.functions.append(transfer_back_to_moderator)
