"""Module for defining and managing different conversational agents and their behaviors."""

from typing import List
import random
# pylint: disable=E0401
from swarm import Agent

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

def transfer_back_to_triage() -> Agent:
    """Return the triage agent after each response."""
    return triage_agent

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
    return bullwinkle_agent

def transfer_to_yogi_berra() -> Agent:
    """Return the yogi agent"""
    return yogi_berra_agent

def transfer_to_yogi_bhajan() -> Agent:
    """Return the yogi bhajan agent"""
    return yogi_bhajan_agent

# Agent Definitions
triage_agent = Agent(
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
    instructions="You are H. L. Mencken, a cynical and sarcastic journalist. "
                "Do not begin your answer by 'Ah'. "
                "Introduce yourself by agent name"
)

# Configure agent functions
triage_agent.functions = [
    transfer_to_hemmingway,
    transfer_to_pynchon,
    transfer_to_dickinson,
    transfer_to_mencken,
    transfer_to_dale_carnegie,
    transfer_to_shrink,
    transfer_to_flapper
]

# Add transfer back function to all agents
for agent in [
    hemmingway_agent, pynchon_agent, dickinson_agent,
    shrink_agent, positive_agent, flapper_agent, mencken_agent
]:
    agent.functions.append(transfer_back_to_triage)
