from swarm import Agent
import random

authors = ["Hemmingway",
           "Pynchon",
           "Emily Dickenson",
           "Dale Carnegie",
           "A Freudian Psychoanalyst",
           "A flapper from the 1920s"]

def get_author():
    author = authors[random.randint(0,len(authors) - 1)]
    print(author)
    return author

triage_agent = Agent(
    name="Triage Agent",
    instructions=f"Transfer to agent whose name == {get_author()}. Call this function after that agent's response",
)

hemmingway_agent = Agent(
    name="Hemmingway",
    instructions=f"""Answer as Hemmingway. Do not begin your answer with 'Ah'.
                     Introduce yourself by agent name"""
)

pynchon_agent = Agent(
    name="Pynchon",
    instructions=f"""Answer as Pynchon. Do not begin your answer with 'Ah'.
                  Introduce yourself by agent name"""

)

dickinson_agent = Agent(
    name="Emily Dickenson",
    instructions=f"""Answer as Emily Dickenson. Do not begin your answer with 'Ah'.
                     Introduce yourself by agent name"""

)
positive_agent = Agent(

    name="Dale Carnegie",
    instructions=f"""Answer as Dale Carnegie. Do not begin your answer with 'Ah'.
                     Introduce yourself by agent name"""

)
shrink_agent = Agent(
    name="A Freudian Psychoanalyst",
    instructions=f"""Answer as A Freudian Psychoanalyst. Do not begin your answer with 'Ah'.
                     Introduce yourself by agent name"""

)
flapper_agent = Agent(
    name="A flapper from the 1920s",
    instructions=f"""Answer as A Flapper from the 1920s. Do not begin your answer with 'Ah'.
                     Introduce yourself by agent name"""

)

def transfer_back_to_triage():
    """call this function after each response."""
    print("In triage")
    return triage_agent


def transfer_to_hemmingway():
    print("transferring to hemmingway")
    return hemmingway_agent

def transfer_to_pynchon():
    print("transferring to pynchon")
    return pynchon_agent

def transfer_to_dickinson():
    print("transferring to dickinson")
    return dickinson_agent

def transfer_to_dale_carnegie():
    print("transferring to positive")
    return positive_agent

def transfer_to_shrink():
    print("transferring to shrink")
    return shrink_agent

def transfer_to_flapper():
    print("transferring to flapper")
    return flapper_agent

triage_agent.functions = [transfer_to_hemmingway,
                          transfer_to_pynchon,
                          transfer_to_dickinson,
                          transfer_to_dale_carnegie,
                          transfer_to_shrink,
                          transfer_to_flapper]

hemmingway_agent.functions.append(transfer_back_to_triage)
pynchon_agent.functions.append(transfer_back_to_triage)
dickinson_agent.functions.append(transfer_back_to_triage)
shrink_agent.functions.append(transfer_back_to_triage)
positive_agent.functions.append(transfer_back_to_triage)
flapper_agent.functions.append(transfer_back_to_triage)

