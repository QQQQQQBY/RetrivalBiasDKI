import requests

from transformers import AutoTokenizer
import os

def call_llm_api_online(prompt=None,
                       messages=None,
                       api_base: str = "LOCAL_API_BASE",
                       model: str = "LOCAL_MODEL",
                       temperature: float = 0.0,
                       api_key: str | None = None,
                       ) -> str:
    url = f"{api_base}/chat/completions"
    if messages is None:
        assert prompt is not None, "Either prompt or messages must be provided"
        messages = [{"role": "system", "content": (
      "Only output a single JSON object and nothing else. No code, no prose, no markdown.\n"
      "If you violate this, replace your entire reply with exactly: "
      '{"rationale":"FORMAT_ERROR","final":[]}'
    )}, {"role": "user", "content": prompt}]
    if api_key is None:
        api_key = os.getenv("SILICONFLOW_API_KEY")
    if api_key is None and api_base.startswith("https://api."):
        raise RuntimeError("Missing API key. Set SILICONFLOW_API_KEY env or pass api_key=...")
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
   
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=500)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException:
        return None
    

def call_llm_api_local(prompt=None,
                       messages=None,
                       api_base: str = "LOCAL_API_BASE",
                       model: str = "LOCAL_MODEL",
                       temperature: float = 0.0,
                       api_key: str | None = None) -> str:
    url = f"{api_base}/chat/completions"
    if messages is None:
        assert prompt is not None, "Either prompt or messages must be provided"
        messages = [{"role": "system", "content": (
      "Only output a single JSON object and nothing else. No code, no prose, no markdown.\n"
      "If you violate this, replace your entire reply with exactly: "
      '{"rationale":"FORMAT_ERROR","final":[]}'
    )},{"role": "user", "content": prompt}]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


