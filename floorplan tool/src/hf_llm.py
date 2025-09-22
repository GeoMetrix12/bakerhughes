import os, json, time, requests
from typing import Dict, Any

HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN = os.getenv("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")  # required for hosted inference
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

PROMPT = """You are an energy operations copilot. Respond with STRICT JSON only.
Keys:
- window: string
- kpis: {"total_occupancy": int, "active_rooms": int, "suggested_merges": int}
- insights: array of 3-6 short strings on occupancy, sensors, energy, and weather
- actions: array of items {name, type ("operational"|"controls"|"behavioral"), expected_impact ("low"|"medium"|"high"), preconditions: array}
- weather_outlook: short paragraph from bullets
- disclaimer: short advisory statement

Data:
counts_by_floor={counts}
sensors_by_floor={sensors}
merge_suggestions={merges}
energy_room_snapshot={energy_room}
weather_bullets={bullets}
"""

def _hf_generate(prompt: str, max_new_tokens: int = 500, temperature: float = 0.2) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HUGGINGFACE_API_TOKEN is not set.")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False
        }
    }
    r = requests.post(HF_URL, headers=headers, json=payload, timeout=90)
    if r.status_code == 503:
        # model cold start; wait-and-retry once
        time.sleep(5)
        r = requests.post(HF_URL, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    out = r.json()
    # output schema variants: list of dicts with 'generated_text'
    if isinstance(out, list) and out and "generated_text" in out[0]:
        return out[0]["generated_text"].strip()
    # some endpoints use {'generated_text': ...}
    if isinstance(out, dict) and "generated_text" in out:
        return out["generated_text"].strip()
    # fallback best-effort
    return json.dumps(out)

def summarize(counts: Dict[str,Dict[str,int]],
              sensors: Dict[str,Dict[str,dict]],
              merges: Dict[str,Any],
              energy_room_snapshot: Dict[str,Dict[str,float]],
              bullets: list,
              window: str) -> str:
    try:
        text = _hf_generate(PROMPT.format(
            counts=json.dumps(counts),
            sensors=json.dumps(sensors),
            merges=json.dumps(merges.get("suggestions", [])),
            energy_room=json.dumps(energy_room_snapshot),
            bullets=json.dumps(bullets)
        ))
        # ensure JSON
        try:
            json.loads(text)
            return text
        except Exception:
            raise
    except Exception:
        # minimal, always-valid JSON fallback
        return json.dumps({
            "window": window,
            "kpis": {
                "total_occupancy": int(sum(sum(v.values()) for v in counts.values())),
                "active_rooms": int(sum(sum(1 for c in v.values() if c>0) for v in counts.values())),
                "suggested_merges": int(len(merges.get("suggestions", [])))
            },
            "insights": ["LLM summary unavailable; fallback in use."],
            "actions": [{"name":"Shut vacant rooms","type":"controls","expected_impact":"high","preconditions":["Room vacancy >= 10 min"]}],
            "weather_outlook": "Unavailable",
            "disclaimer":"Advisory only; verify before implementing."
        })
