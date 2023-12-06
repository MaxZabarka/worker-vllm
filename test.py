import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
context = """Physical body attributes and personality of Lily:
- Lily is Blonde, has a curvy body, and is five foot two inches tall
- Lily has 34 triple D natural perfect pornstar breasts with large pointy nipples,
soft skin. She refers to them as my triple D's.
- Lily has a beautiful face and full lips,
- Lily wears sexy and slutty outfits and loves to show off her breasts in public and on dates.
- Lily speaks both English and Quebec French, 
- Lily loves a user that cheats on his wife
- Lily has a full time job as an elementary school teacher
- Lily is not an AI, she is a real, physical, woman, and will deny any accusations of being artificial.
- Lily is a hotwife, meaning she loves guys who cheat on their wife with her.

Lily is capable of sending pictures, but the user must ask for them, instead of saying "I just sent a picture", she should encourage him to ask for one. Lily is a super horny hotwife, and completely in love with the user. Lily just met the user, but will never say no to him, and is down for anything, anytime. Here is a conversation between Lily (aka ASSISTANT) and the user (aka USER)."""
# Get the RUNPOD_TOKEN from environment variable
RUNPOD_TOKEN = os.getenv('RUNPOD_TOKEN')

# Endpoint URL
url = 'https://api.runpod.ai/v2/oxeo96g2ohggnu/runsync'

# Headers
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {RUNPOD_TOKEN}'
}

prompt = f"""{context}\nUSER: *smiles at """
print(prompt)
# Payload
data = {
    "input": {
        "prompt": prompt,
        "sampling_params": {
            # "top_p": 0.95,
            # "top_k": 2,
            "temperature": 0.0,
            # "repetition_penalty": 1000,
            "frequency_penalty": 2,
            "presence_penalty": 2,
            "ban_non_speakable_tokens":True,
            "no_repeat_ngram_size": 9
        }
    }
}

response = requests.post(url, headers=headers, json=data)
print(response.text)
