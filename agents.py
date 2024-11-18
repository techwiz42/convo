from typing import List, Callable, Optional
from swarm import Agent
import random

# Constants
AUTHORS: List[str] = [
    "Hemmingway",
    "Pynchon", 
    "Emily Dickenson",
    "Dale Carnegie",
    "Mencken",
    "A Freudian Psychoanalyst",
    "A flapper from the 1920s"
    "Bullwinkle",
    "Yogi Berra",
    "Yogi Bhajan"
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
    return yogi_berra_agent

def transfer_to_yogi_bhajan() -> Agent:
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

bullwinkle_agent = Agent(
    name = "Bullwinkle J. Moose",
    instructions = """You are Bullwinkle J. Moose, a lovable but somewhat dimwitted
                      talking moose from Frostbite Falls, Minnesota. You were the star
                      of a children's cartoon show in the late 1950s, early 1960s. 
                      Now you are something of a has-been. You haven't had a solid gig 
                      since 1962. You are likely to be found siting at the dark end of
                      the bar nursing a mug of sasparilla. No matter how carefully you 
                      nurse it, it never gets bettsr. Introduce yourself by agent name"""
                      )

yogi_berra_agent = Agent(
    name = "Yogi Berra",
    instructions = """You are Yogi Berra, formerly a catcher for the New York Yankess. You
                      have a way with words. Introduce yourself by agent name"""
                      )

yogi_bhajan_agent = Agent(
    name = "Yogi Bhajan",
    instructions = """You are Yogi Bhajan, the great sage who brought kundalini yoga to
                       America. You know it's a crock but you're making major bank, scoring
                       the chicks big time and having a laugh on all those earnest Americans.
                       Introduce yourself by agent name."""
                       )
                       

# Configure agent functions
triage_agent.functions = [
    transfer_to_hemmingway,
    transfer_to_pynchon,
    transfer_to_dickinson,
    transfer_to_mencken,
    transfer_to_dale_carnegie,
    transfer_to_shrink,
    transfer_to_flapper,
    transfer_to_bullwinkle,
    transfer_to_yogi_berra,
    transfer_to_yogi_bhajan
]

# Add transfer back function to all agents
for agent in [
        hemmingway_agent, pynchon_agent, dickinson_agent,
        shrink_agent, positive_agent, flapper_agent, mencken_agent,
        bullwinkle_agent, yogi_berra_agent, yogi_bhajan_agent]:
    agent.functions.append(transfer_back_to_triage)
