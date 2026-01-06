from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vision_agents.core import agents   # ✅ Correct import from official docs
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Store active agents by call_id
active_agents: dict[str, object] = {}


class StartAgentRequest(BaseModel):
    call_id: str
    context: list = []


class StopAgentRequest(BaseModel):
    call_id: str


@app.post("/agent/start")
async def start_agent(request: StartAgentRequest):
    if request.call_id in active_agents:
        return {"success": True, "message": "Agent already active"}

    try:
        agent = agents.VisionAgent(
            stream_api_key=os.getenv("STREAM_API_KEY"),
            stream_secret=os.getenv("STREAM_SECRET_KEY"),
            llm={
                "provider": "openai",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": "gpt-4o"
            },
            tts={
                "provider": "elevenlabs",
                "api_key": os.getenv("ELEVENLABS_API_KEY")
            },
            system_prompt=f"""
You are a smart AI meeting assistant.

Meeting context:
{request.context}

Instructions:
Listen to the meeting and only respond when someone says "Hey assistant".
""".strip()
        )

        # Join the Stream call
        await agent.join_call("default", request.call_id)

        active_agents[request.call_id] = agent

        return {"success": True, "message": "Agent joined call"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/stop")
async def stop_agent(request: StopAgentRequest):
    agent = active_agents.get(request.call_id)

    if not agent:
        return {"success": True, "message": "No active agent found"}

    try:
        await agent.leave_call()
        del active_agents[request.call_id]

        return {"success": True, "message": "Agent left call"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}
