import json
import os
import random

GOLDEN_DATA_PATH = "it-academy-hackathon-solution-example/data/Go Nova.json"
OUTPUT_PATH = "it-academy-hackathon-solution-example/data/stress_test_500.json"
EXPECTED_GOLDEN_IDS = [
    "3666666666666666666", "3888888888888888888", "4111111111111111110", 
    "4222222222222222221", "4444444444444444444", "4555555555555555555", 
    "5111111111111111110", "5222222222222222221", "5444444444444444443", 
    "5555555555555555555", "5888888888888888888", "4999999999999999999", 
    "6222222222222222221", "3777777777777777777", "3999999999999999999", 
    "4666666666666666666", "5999999999999999999"
]

TECH_BLOAT = [
    "def example(): return [i for i in range(100)]",
    "{\"status\": \"ok\", \"logs\": [\"error at line 42\", \"warning: deprecated API\"]}",
    "CI/CD Pipeline #1234: SUCCESS. Steps: Build, Test, Deploy.",
    "SQL: SELECT * FROM users JOIN orders ON users.id = orders.user_id WHERE status = 'active';",
    "package main; import \"fmt\"; func main() { fmt.Println(\"Hello World\") }",
]

DECOY_TEMPLATES = [
    "Go to the kitchen and grab some coffee.",
    "The release of the new movie is scheduled for tomorrow.",
    "I have a meetup with my relatives this weekend.",
    "Let's Go for a walk.",
    "Release the kraken!",
    "Virtual meetup starting in 5 minutes.",
]

CHATTER = [
    "Okay", "Agree", "Yes!", "??", "👍", "Interesting link: http://example.com", "Haha", "Cool",
]

def generate():
    with open(GOLDEN_DATA_PATH, "r") as f:
        data = json.load(f)
    
    golden_messages = [m for m in data["messages"] if m["id"] in EXPECTED_GOLDEN_IDS]
    
    # Add 3 more to make it 20 golden messages
    current_ids = {m["id"] for m in golden_messages}
    other_messages = [m for m in data["messages"] if m["id"] not in current_ids]
    golden_messages.extend(other_messages[:20 - len(golden_messages)])
    
    noisy_messages = []
    
    # 200 Technical Bloat
    for i in range(200):
        noisy_messages.append({
            "id": f"noisy_bloat_{i}",
            "thread_sn": None,
            "time": 1700000000 + i,
            "text": random.choice(TECH_BLOAT) * 5, 
            "sender_id": "system@corp.example",
            "file_snippets": "",
            "parts": [],
            "mentions": [],
            "member_event": None,
            "is_system": True,
            "is_hidden": False,
            "is_forward": False,
            "is_quote": False
        })
    
    # 100 Semantic Decoys
    for i in range(100):
        noisy_messages.append({
            "id": f"noisy_decoy_{i}",
            "thread_sn": None,
            "time": 1700000200 + i,
            "text": random.choice(DECOY_TEMPLATES),
            "sender_id": "user@corp.example",
            "file_snippets": "",
            "parts": [],
            "mentions": [],
            "member_event": None,
            "is_system": False,
            "is_hidden": False,
            "is_forward": False,
            "is_quote": False
        })
        
    # 180 Conversational Chatter
    for i in range(180):
        noisy_messages.append({
            "id": f"noisy_chatter_{i}",
            "thread_sn": None,
            "time": 1700000300 + i,
            "text": random.choice(CHATTER),
            "sender_id": "chatter@corp.example",
            "file_snippets": "",
            "parts": [],
            "mentions": [],
            "member_event": None,
            "is_system": False,
            "is_hidden": False,
            "is_forward": False,
            "is_quote": False
        })
        
    final_messages = golden_messages + noisy_messages
    random.shuffle(final_messages)
    
    data["messages"] = final_messages
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Generated 500 messages in {OUTPUT_PATH}")

if __name__ == "__main__":
    generate()
