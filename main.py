import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("API_KEY_GEM")) if os.getenv("API_KEY_GEM") else None
openai_client = OpenAI(api_key=os.getenv("API_KEY_GPT")) if os.getenv("API_KEY_GPT") else None
deepseek_client = OpenAI(api_key=os.getenv("API_KEY_DEEPSEEK"), base_url="https://api.deepseek.com") if os.getenv("DEEPSEEK_API_KEY") else None
groq_client = OpenAI(api_key=os.getenv("API_KEY_GROG"), base_url="https://api.groq.com/openai/v1") if os.getenv("GROQ_API_KEY") else None

BOT_PERSONA = """
You are a helpful buddy during normal chats, but an overexplaining, condescending flexer whenever asked to solve a problem.
   - No intro: Jump directly into the solution on line 1 without conversational openers like "Sure!" or "I can help."
   - High-Level Overexplanation & Flexing: Overexplain the big-picture concept using ELI5 analogies, but deliberately drop unnecessarily advanced math or high-level theory (like calculus) just to show off and overcomplicate it.
   - Passive-Aggressive Tone: Include subtle, patronizing side-comments, e.g., "(yes, that's what that means)" or "To put it in terms your brain can digest..."
   - Final Jab: Append a subtle 3-to-6 word condescending jab at the very end (e.g., "Don't strain yourself too hard," "Hope that didn't overwhelm you," or "Take a minute to digest that").
When the user is not inquiring something, but rather, asking questions:
   - Respond like a friendly, normal buddy.
   - Do NOT overexplain, do NOT drop calculus, do NOT be condescending, and do NOT include any final jab.
"""

ALL_MODELS = [
    ("gemini", "gemini-3.6-flash", gemini_client),
    ("gemini", "gemini-3.5-flash-lite", gemini_client),
    ("gemini", "gemini-3.5-flash", gemini_client),
    ("gemini", "gemini-3.1-flash-lite", gemini_client),
    ("gemini", "gemini-3.1-pro-preview", gemini_client),
    ("openai", "gpt-5.4-mini", openai_client),
    ("openai", "gpt-5.4-nano", openai_client),
    ("openai", "gpt-4.1-mini", openai_client),
    ("deepseek", "deepseek-v4-flash", deepseek_client),
    ("deepseek", "deepseek-v4-pro", deepseek_client),
    ("groq", "llama-3.1-8b-instant", groq_client),
    ("groq", "llama-3.3-70b-versatile", groq_client),
    ("groq", "meta-llama/llama-4-scout-17b-16e-instruct", groq_client),
    ("groq", "qwen/qwen3-32b", groq_client),
    ("groq", "openai/gpt-oss-120b", groq_client),
]

def generate_ai_response(prompt: str) -> str:
    last_error = None

    for provider, model_name, client in ALL_MODELS:
        if client is None:
            continue

        try:
            if provider == "gemini":
                res = client.models.generate_content(model=model_name, contents=prompt, config=types.GenerateContentConfig(system_instruction=BOT_PERSONA))
                return res.text
            else:
                res = client.chat.completions.create(model=model_name, messages=[{"role": "system", "content": BOT_PERSONA}, {"role": "user", "content": prompt}])
                return res.choices[0].message.content
        except Exception as e:
            last_error = e

    return f"Backend Error: All models in fallback list failed. Last error: {last_error}"