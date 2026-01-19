import pandas as pd

df = pd.read_parquet("train-00000-of-00001.parquet")

df = df.groupby('claim_id').first().reset_index()

TAXONOMY = {
    "0_0": "No disinformation narrative",
    "1_0": "Global warming is not happening",
    "1_1": "Ice/permafrost/snow cover isn't melting",
    "1_2": "We're heading into an ice age/global cooling",
    "1_3": "Weather is cold/snowing",
    "1_4": "Climate hasn't warmed/changed over the last (few) decade(s)",
    "1_5": "Oceans are cooling/not warming",
    "1_6": "Sea level rise is exaggerated/not accelerating",
    "1_7": "Extreme weather isn't increasing/has happened before/isn't linked to climate change",
    "1_8": "They changed the name from 'global warming' to 'climate change'",
    "2_0": "Human greenhouse gases are not causing climate change",
    "2_1": "It's natural cycles/variation",
    "2_2": "It's non-greenhouse gas human climate forcings (aerosols, land use)",
    "2_3": "There's no evidence for greenhouse effect/carbon dioxide driving climate change",
    "2_4": "CO2 is not rising/ocean pH is not falling",
    "2_5": "Human CO2 emissions are miniscule/not raising atmospheric CO2",
    "3_0": "Climate impacts/global warming is beneficial/not bad",
    "3_1": "Climate sensitivity is low/negative feedbacks reduce warming",
    "3_2": "Species/plants/reefs aren't showing climate impacts yet/are benefiting from climate change",
    "3_3": "CO2 is beneficial/not a pollutant",
    "3_4": "It's only a few degrees (or less)",
    "3_5": "Climate change does not contribute to human conflict/threaten national security",
    "3_6": "Climate change doesn't negatively impact health",
    "4_0": "Climate solutions won't work",
    "4_1": "Climate policies (mitigation or adaptation) are harmful",
    "4_2": "Climate policies are ineffective/flawed",
    "4_3": "It's too hard to solve",
    "4_4": "Clean energy technology/biofuels won't work",
    "4_5": "People need energy (e_g_, from fossil fuels/nuclear)",
    "5_0": "Climate movement/science is unreliable",
    "5_1": "Climate-related science is uncertain/unsound/unreliable (data, methods & models)",
    "5_2": "Climate movement is alarmist/wrong/political/biased/hypocritical (people or groups)",
    "5_3": "Climate change (science or policy) is a conspiracy (deception)"
}

taxonomy_str = "\n".join([f"{k}: {v}" for k, v in TAXONOMY.items()])

def build_prompt(claim):
    return f"""You are an expert in detecting climate change related disinformation. You get a claim and your task is to classify the claim using the taxonomy codes provided. 
Rules:
- If the claim aligns with narratives from the taxonomy, list applicable codes separated by a semicolon (;).
- If no disinformation is found, return '0_0'.

Taxonomy:
{taxonomy_str}

Claim: "{claim}"
Codes:"""

claims = df["claim"].tolist()
narratives = df["narrative"].tolist()

messages_all = []
import json
for i in range(len(claims)):
    print(i)
    claim = claims[i]
    codes = narratives[i]

    prompt = build_prompt(claim)

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": codes}
    ]

    messages_all.append({"messages": messages})

with open("cc_messages.jsonl", 'w', encoding='utf-8') as f:
    for message in messages_all:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")