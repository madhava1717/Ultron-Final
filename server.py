import json
import os
import asyncio
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from duckduckgo_search import DDGS
import uvicorn

app = FastAPI(title="Ultron Global AI Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Yahan apni working Groq API Key daalein
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

def search_web(query: str) -> str:
    """Internet se clean information fetch karne ka function."""
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return ""
        return "\n".join([f"- {res.get('title', '')}: {res.get('body', '')}" for res in results])
    except Exception as e:
        return ""

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for latest breaking news, real-time sports scores, weather, or recent events. Do NOT use for basic general knowledge, history, current prime minister/presidents, or time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Concise search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WEBSOCKET CONNECTED] Global AI link established.")
    history = []

    try:
        while True:
            data = await websocket.receive_text()
            if not data.strip():
                continue

            current_time = datetime.now().strftime("%I:%M %p, %A, %d %B %Y")
            system_prompt = {
                "role": "system",
                "content": (
                    f"You are Ultron, an ultra-intelligent, sharp, and natural voice assistant. "
                    f"The current local date and time is {current_time}. "
                    "Always give concise, direct, spoken-style responses (1-3 sentences max). "
                    "Never say 'Based on the search results', 'According to excerpts', or 'As an AI'. "
                    "Answer directly like Jarvis or a real human peer."
                )
            }

            if not history:
                history.append(system_prompt)
            else:
                history[0] = system_prompt

            history.append({"role": "user", "content": data})
            print(f"[USER]: {data}")

            try:
                response = await groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=history,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=300
                )

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                if tool_calls:
                    search_context = ""
                    for tool_call in tool_calls:
                        if tool_call.function.name == "search_web":
                            try:
                                args = json.loads(tool_call.function.arguments)
                                query = args.get("query", data)
                            except:
                                query = data
                            print(f"[SYSTEM] Live web lookup for: {query}")
                            search_context = await asyncio.to_thread(search_web, query)

                    # Model ko natural synthesis ke liye augmented prompt dena
                    augmented_messages = history.copy()
                    if search_context:
                        augmented_messages.append({
                            "role": "system",
                            "content": f"Live Web Information:\n{search_context}\n\nCombine this information with your general knowledge to give a direct, natural answer to the user's question. Do not mention search or sources."
                        })

                    final_response = await groq_client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=augmented_messages,
                        max_tokens=300
                    )
                    reply = final_response.choices[0].message.content.strip()
                else:
                    reply = response_message.content.strip()

                history.append({"role": "assistant", "content": reply})

                if len(history) > 12:
                    history = [history[0]] + history[-10:]

                print(f"[ULTRON]: {reply}")
                await websocket.send_text(reply)

            except Exception as e:
                print(f"[API ERROR]: {e}")
                await websocket.send_text("I encountered a glitch retrieving that. Please ask again.")

    except WebSocketDisconnect:
        print("[WEBSOCKET DISCONNECTED] Client closed connection.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)