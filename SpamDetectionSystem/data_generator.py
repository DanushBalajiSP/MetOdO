import pandas as pd
import random

def generate_data(num_samples=200):
    spam_templates = [
        "Win a free iPhone now! Click here: {url}",
        "Congratulations! You've won a lottery of ${amount}. Claim at {url}",
        "Cheap meds! Viagra, Cialis. Visit {url}",
        "Hot singles in your area! Chat now {url}",
        "Urgent! Your account has been compromised. Verify at {url}",
        "Make money from home! Earn ${amount}/day. Sign up {url}",
        "Free crypto giveaway! Bitcoin, Ethereum. Join {url}"
    ]

    ham_templates = [
        "Just had a great lunch at {place}.",
        "Can't believe it's already {day}!",
        "Watching the game tonight. Go team!",
        "Anyone know a good place for {activity}?",
        "So tired from work today. Need sleep.",
        "Listening to my favorite song.",
        "Happy birthday to my best friend!",
        "Learning machine learning is fun but hard.",
        "Does anyone have notes for the {subject} class?",
        "Beautiful sunset today!"
    ]

    urls = ["http://bit.ly/fake", "www.scam.com", "http://phony.site", "www.free-prizes.net"]
    amounts = ["1000", "5000", "10000", "500"]
    places = ["Burger King", "the park", "my house", "downtown"]
    days = ["Monday", "Friday", "Sunday"]
    activities = ["hiking", "sushi", "coffee"]
    subjects = ["math", "history", "science"]

    data = []

    for _ in range(num_samples // 2):
        # Generate Spam
        template = random.choice(spam_templates)
        text = template.format(url=random.choice(urls), amount=random.choice(amounts))
        data.append({"text": text, "label": "spam"})

        # Generate Ham
        template = random.choice(ham_templates)
        text = template.format(place=random.choice(places), day=random.choice(days), 
                               activity=random.choice(activities), subject=random.choice(subjects))
        data.append({"text": text, "label": "ham"})

    df = pd.DataFrame(data)
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv("spam_data.csv", index=False)
    print(f"Generated {len(df)} samples in spam_data.csv")

if __name__ == "__main__":
    generate_data(500)
