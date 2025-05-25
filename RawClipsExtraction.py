import pandas as pd
import yt_dlp
import time

BASE_KEYWORDS = ["squat form",
    "how to squat",
    "barbell squat",
    "front squat",
    "back squat",
    "squat tutorial",
    "squat side view",
    "squat exercise",
    "perfect squat technique",
    "squat form check",
    "pullup form",
    "how to do a pullup",
    "pullups tutorial",
    "pullup slow motion",
    "strict pullup",
    "chin up vs pull up",
    "perfect pullup technique",
    "pullup progression",
    "pullup exercise",
    "pullup form check",
    "deadlift form",
    "how to deadlift",
    "deadlift tutorial",
    "romanian deadlift",
    "sumo deadlift",
    "deadlift side view",
    "perfect deadlift technique",
    "deadlift slow motion",
    "deadlift exercise",
    "deadlift form check"
]


# Youtube search usign yt_dlp
def youtube_search(query, max_results=15):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'forcejson': True,
        'simulate': True,
        'default_search': 'ytsearch',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        results = ydl.extract_info(f"ytsearch25:{query}", download=False)
    
    video_data = []
    for entry in results.get('entries', []):
        duration = entry.get('duration', 0)
        if(duration and duration<=180):
            video_data.append({
                'title': entry['title'],
                'url': f"https://www.youtube.com/watch?v={entry['id']}"
            })
        if len(video_data) >= max_results:
            break
    return video_data

LABEL_KEYWORDS = {
    "Squat": ["squat", "squats", "back squat", "front squat", "squat form", "squat technique"],
    "Pull-Up": ["pullup", "pull-up", "pull ups", "pull up", "chin up", "chin-up", "strict pullup", "pullup form"],
    "Deadlift": ["deadlift", "deadlifts", "romanian deadlift", "sumo deadlift", "conventional deadlift", "deadlift form"]
}

def assign_label(title):
    title = title.lower()
    for label, phrases in LABEL_KEYWORDS.items():
        for phrase in phrases:
            if phrase in title:
                return label
    return "unknown"

# search and save
def search_and_save(base_keywords, output_csv="youtube_scraped.csv", results_per_query=15):
    all_entries = []

    for query in base_keywords:
        print(f"🔍 Searching for: {query}")
        try:
            videos = youtube_search(query, max_results=results_per_query)
            for video in videos:
                label = assign_label(video['title'])
                print(f"  {video['title']}")
                all_entries.append({
                    "title": video['title'],
                    "url": video['url'],
                    "label": label
                })
        except Exception as e:
            print(f" Failed search for: {query}\n{e}")
        time.sleep(1)  # Being polite to youtube

    # Remove duplicates and save
    df = pd.DataFrame(all_entries).drop_duplicates(subset='url')
    df.to_csv(output_csv, index=False)
    print(f"\n Saved {len(df)} results to {output_csv}")

# Run
search_and_save(BASE_KEYWORDS)
