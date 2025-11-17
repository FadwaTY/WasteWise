import os, json, requests
from typing import Dict, List, Optional

# Lightweight domain tips used as grounding context + fallback
DOMAIN_TIPS = {
    "PAPER": [
        "Keep paper dry and free of food residue.",
        "Remove tape/staples when possible; glossy/laminated paper may not be recyclable."
    ],
    "CARDBOARD": [
        "Flatten boxes to save space; remove plastic liners or labels.",
        "Greasy pizza boxes: only the clean parts should be recycled."
    ],
    "PLASTIC": [
        "Rinse containers; check local rules for resin codes; avoid film/wrap in commingled bins.",
        "Squeeze air out; keep caps on only if accepted in your area."
    ],
    "GLASS": [
        "Rinse bottles/jars; remove caps; avoid mixing broken glass unless your program accepts it.",
        "Separate by color if required locally."
    ],
    "METAL": [
        "Rinse cans; crush lightly if space constrained.",
        "Clean aluminum foil can be balled up to the size of a fist for easier sorting."
    ],
    "BIODEGRADABLE": [
        "Compost food scraps and yard waste; avoid plastic contamination.",
        "Keep meat/dairy out unless your composting facility accepts them."
    ],
}

def _build_prompt(counts: Dict[str, int], detections: List[Dict]) -> str:
    present = [k.upper() for k in counts.keys()]
    ctx_lines = []
    for k in present:
        for tip in DOMAIN_TIPS.get(k, []):
            ctx_lines.append(f"- {k}: {tip}")
    inp = {"counts": counts, "detections": [
        {"class_name": d.get("class_name"), "confidence": float(d.get("confidence", 0.0))}
        for d in detections
    ]}
    return f"""You are WasteWise, a recycling & waste-sorting assistant.
                Use the context tips to craft concise, safe, generic (non-location-specific) guidance.
                Avoid legal claims. Prefer actionable bullets. If unsure, say so.

                Context tips:
                {os.linesep.join(ctx_lines) or "(none)"}

                Input JSON:
                ```json
                {json.dumps(inp, ensure_ascii=False)}
                Write Markdown with sections:

                Summary (1 line referencing totals)

                Sort & Dispose (3-5 bullets)

                Contamination Watchouts (2-4 bullets)

                Sustainability Fact (1 line)

                Next Actions (1-2 bullets)
                Keep under 120 words.
                """

def _fallback_text(counts: Dict[str,int]) -> str:
    total = sum(counts.values())
    parts = [f"Summary: {total} item(s) - " + ", ".join(f"{v} {k}" for k, v in counts.items())]
    tips = []
    for k, v in counts.items():
        tip = DOMAIN_TIPS.get(k.upper(), ["Handle responsibly."])[0]
        tips.append(f"- {k}: {tip}")
        parts.append("Sort & Dispose:\n" + "\n".join(tips[:5]))
        parts.append("Sustainability Fact: Recycling right reduces contamination and improves material recovery.")
    return "\n\n".join(parts)

def recommend(counts: Dict[str,int], detections: List[Dict]) -> Optional[str]:
    provider = os.getenv("WW_LLM_PROVIDER", "ollama").lower() # 'ollama' | 'llamacpp' | 'openai' | 'none'
    if provider in ("none", "", "false"):
        return None

    prompt = _build_prompt(counts, detections)
    try:
        if provider == "ollama":
            url   = os.getenv("WW_OLLAMA_URL", "http://localhost:11434")
            model = os.getenv("WW_LLM_MODEL", "llama3.2")
            r = requests.post(f"{url}/api/generate",
                            json={"model": model, "prompt": prompt,
                                  "stream": False, "keep_alive":"10m",
                                  "options": {
                                      "temperature": 0.2,
                                      "num_predict":150
                                      }
                                  },
                            timeout=120)
            r.raise_for_status()
            return r.json().get("response", "").strip()

        elif provider == "llamacpp":
            from llama_cpp import Llama  # optional dependency
            model_path = os.getenv("WW_LLAMA_PATH", "models/llm.gguf")
            llm = Llama(model_path=model_path, n_ctx=4096)
            out = llm(prompt, max_tokens=256, temperature=0.2)
            return out["choices"][0]["text"].strip()

        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI()  # needs OPENAI_API_KEY
            model = os.getenv("WW_LLM_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are WasteWise assistant."},
                        {"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=256
            )
            return resp.choices[0].message.content.strip()

    except Exception:
        pass  # fall back if anything goes wrong

    return _fallback_text(counts)
