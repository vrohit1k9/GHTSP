"""
Generates synthetic tweet dataset (~2,000 tweets over 2 days: 2012-10-11 and 2012-10-12)
with planted events:
1. Vice Presidential Debate (#VpDebate, #Oct12Debate, Joe Biden, Paul Ryan, foreign policy, medicare)
2. National Coming Out Day (#nationalcomingoutday, #BreakingNews, equality, pride, human rights)
3. Felix Baumgartner Space Jump / Mars Rover background event
4. Everyday conversational noise
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = DATA_DIR / "synthetic_tweets.csv"
ANNOTATION_CSV = DATA_DIR / "ground_truth_annotations.csv"

# Time boundaries: Oct 11 00:00:00 to Oct 12 23:59:59 (48 hours = 2 x 24h windows)
START_TIME = datetime(2012, 10, 11, 0, 0, 0)
TOTAL_SECONDS = 48 * 3600

USERS = [
    f"user_{i}" for i in range(1, 401)
] + [
    "political_junkie", "news_wire", "debate_watcher", "equality_now", "tech_insider",
    "daily_buzz", "policy_wonk", "campaign_trail", "live_tweeter", "campus_voice"
]

# --- Planted Event 1: VP Debate (heavy on evening of Oct 11 & morning of Oct 12) ---
DEBATE_TEMPLATES = [
    "Watching @joebiden and @paulryan clash over foreign policy and medicare in Kentucky! #VpDebate #Oct12Debate http://t.co/deb891",
    "Intense exchange on the jobs report and economy tonight between Joe Biden and Paul Ryan. #VpDebate",
    "Martha Raddatz is pressing hard on middle east foreign policy and libya. #VpDebate #BreakingNews",
    "Who is winning the vice presidential debate? Poll numbers showing close race between Biden and Ryan. #Oct12Debate",
    "Debate fact check: Joe Biden defends medicare and social security while Paul Ryan argues for tax reform. #VpDebate",
    "Unbelievable clash over foreign policy and middle east strategy in tonight's debate! #Oct12Debate #VpDebate",
    "@joebiden brings energetic defense on middle east diplomacy and jobs report. #VpDebate",
    "@paulryan highlights economy and national debt concerns during the debate. #Oct12Debate",
    "Post debate poll instant reaction: Undecided voters split on vice presidential debate performance. #VpDebate",
    "The vice presidential debate in Danville Kentucky is heating up on foreign policy and taxes! #VpDebate #BreakingNews",
    "Both candidates arguing passionately about foreign policy and democratic party platform. #Oct12Debate",
    "Analysis of the vice presidential debate: clash over medicare and budget deficit. #VpDebate",
    "Vice presidential debate replay: key moments between Joe Biden and Paul Ryan on middle east. #Oct12Debate",
    "Watch the full vice presidential debate live stream right now! #VpDebate http://t.co/stream12"
]

# --- Planted Event 2: National Coming Out Day (peaks Oct 11) ---
NCOD_TEMPLATES = [
    "Celebrating courage and equality today across the country! #nationalcomingoutday #love",
    "Proud to stand with the lgbt community for human rights and equality! #nationalcomingoutday #BreakingNews",
    "Everyone deserves dignity, acceptance, and civil rights. Happy #nationalcomingoutday to all!",
    "Inspiring stories of courage and pride shared today nationwide. #nationalcomingoutday #equality",
    "Support equality and equal rights for everyone in our community! #nationalcomingoutday #pride",
    "Standing together for diversity, inclusion, and love is love! #nationalcomingoutday #humanrights",
    "Celebrating #nationalcomingoutday with courage and community support! #pride",
    "Honoring historic milestones in the civil rights movement today. #nationalcomingoutday #BreakingNews",
    "Youth and allies coming together for acceptance and hope! #nationalcomingoutday",
    "Happy #nationalcomingoutday! Let us continue the fight for equal rights and marriage equality."
]

# --- Planted Event 3: Red Bull Stratos Space Jump / Curiosity Rover ---
SPACE_TEMPLATES = [
    "Felix Baumgartner preparing for the historic supersonic space jump with Red Bull Stratos! #BreakingNews",
    "NASA Curiosity rover sends back stunning new images and science data from Mars surface.",
    "Watching the Red Bull Stratos space jump broadcast live! Incredible milestone in science and technology.",
    "Curiosity rover analyzes soil samples on Mars, revealing new details about planetary discovery."
]

# --- Noise Templates ---
NOISE_TEMPLATES = [
    "Drinking my morning coffee and heading to work. Traffic on the bridge is terrible today.",
    "Just finished a great workout at the gym. Feeling refreshed and ready for the weekend!",
    "Anyone watching the football game tonight? That touchdown was incredible!",
    "Craving pizza for lunch. Best slice in town or making one at home?",
    "Finally finished reading this amazing book. Highly recommend it to everyone!",
    "Rainy weather outside today. Perfect time to stay in and watch movies with tea.",
    "Working late on this software project. Debugging code can be so satisfying when it works.",
    "Beautiful sunset over the park this evening. Autumn colors are gorgeous.",
    "Listening to my favorite music playlist on repeat while studying.",
    "Can't believe it's already Friday tomorrow. This week went by so fast!",
    "Checked out the new farmers market downtown. Fresh apples and cider!",
    "My cat is sleeping on my laptop keyboard again. Guess no work for me.",
    "Flight delayed at the airport. Catching up on emails and podcasts.",
    "Delicious homemade dinner tonight with pasta and garlic bread.",
    "Looking forward to the weekend hiking trip in the mountains."
]


def generate_dataset(total_tweets=2100):
    rows = []
    
    # 1. Generate ~650 VP Debate tweets (concentrated around Oct 11 18:00 to Oct 12 12:00)
    for i in range(650):
        tid = f"tweet_{100000 + len(rows)}"
        user = random.choice(USERS)
        text = random.choice(DEBATE_TEMPLATES)
        # Random variation to text
        if random.random() < 0.3:
            text = text + " " + random.choice(["So intense!", "Must watch.", "Thoughts?", "Agree or disagree?"])
        # Concentrated time: Oct 11 18:00 (sec 64800) to Oct 12 14:00 (sec 136800)
        sec = int(random.gauss(85000, 18000))
        sec = max(3600, min(TOTAL_SECONDS - 60, sec))
        dt = START_TIME + timedelta(seconds=sec)
        
        rt = int(random.paretovariate(1.4) * 8)
        followers = int(random.paretovariate(1.2) * 250) + 50
        
        rows.append({
            "id": tid,
            "user": user,
            "text": text,
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "retweet_count": rt,
            "follower_count": followers,
            "_cluster_label": "debate"
        })

    # 2. Generate ~350 National Coming Out Day tweets (concentrated on Oct 11 08:00 to 22:00)
    for i in range(350):
        tid = f"tweet_{100000 + len(rows)}"
        user = random.choice(USERS)
        text = random.choice(NCOD_TEMPLATES)
        if random.random() < 0.25:
            text = text + " " + random.choice(["Incredible courage!", "Proud!", "Stand strong."])
        # Concentrated time: Oct 11 08:00 (sec 28800) to 22:00 (sec 79200)
        sec = int(random.gauss(50000, 15000))
        sec = max(60, min(86000, sec))
        dt = START_TIME + timedelta(seconds=sec)
        
        rt = int(random.paretovariate(1.5) * 5)
        followers = int(random.paretovariate(1.3) * 180) + 40
        
        rows.append({
            "id": tid,
            "user": user,
            "text": text,
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "retweet_count": rt,
            "follower_count": followers,
            "_cluster_label": "ncod"
        })

    # 3. Generate ~100 Space / Science tweets
    for i in range(100):
        tid = f"tweet_{100000 + len(rows)}"
        user = random.choice(USERS)
        text = random.choice(SPACE_TEMPLATES)
        sec = random.randint(100, TOTAL_SECONDS - 100)
        dt = START_TIME + timedelta(seconds=sec)
        rt = random.randint(1, 20)
        followers = random.randint(100, 3000)
        rows.append({
            "id": tid,
            "user": user,
            "text": text,
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "retweet_count": rt,
            "follower_count": followers,
            "_cluster_label": "science"
        })

    # 4. Generate ~1,000 noise tweets spread across the 48 hours
    while len(rows) < total_tweets:
        tid = f"tweet_{100000 + len(rows)}"
        user = random.choice(USERS)
        text = random.choice(NOISE_TEMPLATES)
        # random hashtags or mentions in noise
        if random.random() < 0.15:
            text += random.choice([" #daily", " #life", " #random", " #fun", " @friend"])
        sec = random.randint(0, TOTAL_SECONDS - 1)
        dt = START_TIME + timedelta(seconds=sec)
        rt = int(random.paretovariate(2.5) * 2)
        followers = int(random.paretovariate(1.5) * 50) + 10
        rows.append({
            "id": tid,
            "user": user,
            "text": text,
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "retweet_count": rt,
            "follower_count": followers,
            "_cluster_label": "noise"
        })

    # Sort tweets chronologically
    rows.sort(key=lambda r: r["timestamp"])
    
    # Save synthetic_tweets.csv (without internal label)
    fieldnames = ["id", "user", "text", "timestamp", "retweet_count", "follower_count"]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
            
    print(f"Generated {len(rows)} tweets in {OUTPUT_CSV}")
    
    # Generate ground truth annotation CSV
    # format: cluster_id, is_realistic_event, event_id
    # Used by evaluate.py to score clusters against realistic events
    annotations = [
        {"cluster_id": "cluster_vp_debate", "is_realistic_event": 1, "event_id": "event_vp_debate_2012"},
        {"cluster_id": "cluster_biden_ryan", "is_realistic_event": 1, "event_id": "event_vp_debate_2012"},
        {"cluster_id": "cluster_ncod", "is_realistic_event": 1, "event_id": "event_ncod_2012"},
        {"cluster_id": "cluster_space_jump", "is_realistic_event": 1, "event_id": "event_stratos_2012"},
        {"cluster_id": "cluster_noise_1", "is_realistic_event": 0, "event_id": ""},
        {"cluster_id": "cluster_noise_2", "is_realistic_event": 0, "event_id": ""},
    ]
    with open(ANNOTATION_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster_id", "is_realistic_event", "event_id"])
        writer.writeheader()
        for a in annotations:
            writer.writerow(a)
    print(f"Generated annotations in {ANNOTATION_CSV}")


if __name__ == "__main__":
    generate_dataset()
