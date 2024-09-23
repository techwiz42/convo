import json
import random
from typing import List, Dict

def generate_dataset(num_entries: int = 5000) -> List[Dict[str, str]]:
    topics = [
        "history", "science", "literature", "geography", "technology",
        "arts", "sports", "politics", "economics", "philosophy",
        "music", "cinema", "biology", "physics", "mathematics"
    ]
    
    templates = [
        {
            "context": "{subject} is {description}. It {fact1}. Additionally, {fact2}.",
            "question": "What is {question_focus} of {subject}?",
            "answer": "{answer}"
        },
        {
            "context": "In the field of {field}, {subject} is known for {achievement}. This {impact} and led to {consequence}.",
            "question": "What was the main achievement of {subject} in {field}?",
            "answer": "{achievement}"
        },
        {
            "context": "The {event} occurred in {year}. It was characterized by {characteristic} and resulted in {result}.",
            "question": "When did the {event} take place?",
            "answer": "{year}"
        },
        {
            "context": "{location} is famous for its {feature}. It attracts {visitors} annually and is known for {attraction}.",
            "question": "What is {location} famous for?",
            "answer": "Its {feature}"
        },
        {
            "context": "In {year}, {person} made a significant discovery in {field}. This discovery {impact} and opened new avenues for {application}.",
            "question": "What field did {person} make a significant discovery in?",
            "answer": "{field}"
        }
    ]
    
    data = []
    
    for _ in range(num_entries):
        template = random.choice(templates)
        topic = random.choice(topics)
        
        entry = {
            "context": template["context"].format(
                subject=f"The {topic} of {random.choice(['ancient', 'modern', 'contemporary', 'classical'])} times",
                description=random.choice(["a fascinating area of study", "a complex field", "an evolving discipline"]),
                fact1=f"has roots dating back to {random.randint(1000, 2000)} AD",
                fact2=f"has influenced {random.choice(['culture', 'science', 'politics', 'art'])} significantly",
                field=topic,
                achievement=f"the development of {random.choice(['theory', 'technology', 'methodology'])}",
                impact=f"revolutionized {random.choice(['thinking', 'practice', 'understanding'])} in the field",
                consequence=f"numerous advancements in {random.choice(['research', 'application', 'theory'])}",
                event=f"The Great {topic.capitalize()} Revolution",
                year=random.randint(1700, 2020),
                characteristic=random.choice(["rapid change", "social upheaval", "technological advancement"]),
                result=f"a paradigm shift in {topic}",
                location=f"The {random.choice(['City', 'Region', 'Country'])} of {topic.capitalize()}",
                feature=random.choice(["historical significance", "natural beauty", "cultural heritage"]),
                visitors=f"{random.randint(1, 10)} million visitors",
                attraction=f"its unique {random.choice(['architecture', 'customs', 'natural wonders'])}",
                person=f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez'])}",
                application=f"research in {random.choice(['medicine', 'technology', 'environmental science'])}"
            ),
            "question": template["question"].format(
                question_focus=random.choice(["the main characteristic", "the historical significance", "the primary impact"]),
                subject=topic,
                event=f"The Great {topic.capitalize()} Revolution",
                location=f"The {random.choice(['City', 'Region', 'Country'])} of {topic.capitalize()}",
                person=f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez'])}"
            ),
            "answer": template["answer"].format(
                answer=random.choice(["Its historical significance", "Its impact on modern society", "Its role in shaping culture"]),
                achievement=f"the development of {random.choice(['theory', 'technology', 'methodology'])}",
                year=random.randint(1700, 2020),
                feature=random.choice(["historical significance", "natural beauty", "cultural heritage"]),
                field=topic
            )
        }
        
        data.append(entry)
    
    return data

# Generate the dataset
dataset = generate_dataset(5000)

# Save the dataset to a JSON file
with open('data/large_training_dataset.json', 'w') as f:
    json.dump({"data": dataset}, f, indent=2)

print("Dataset with 5,000 entries has been generated and saved to 'data/large_training_dataset.json'")
