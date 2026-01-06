from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vision_agents import VisionAgent
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Store active agents
active_agents = {}

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
        # Initialize Vision Agent
        agent = VisionAgent(
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
            system_prompt=f"""You are a meeting assistant.
            
Context: {request.context}

Listen and answer when someone says 'Hey assistant'."""
        )
        
        # Join the Stream Video call
        await agent.join_call("default", request.call_id)
        
        # Store agent
        active_agents[request.call_id] = agent
        
        return {"success": True, "message": "Agent joined"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/stop")
async def stop_agent(request: StopAgentRequest):
    agent = active_agents.get(request.call_id)
    
    if agent:
        await agent.leave_call()
        del active_agents[request.call_id]
    
    return {"success": True, "message": "Agent left"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
