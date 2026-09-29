from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from agents import run_research_crew
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

app = FastAPI(title="Multi-Agent Orchestrator API")

# request aur response ke liye Pydantic models define kiye gaye hain    
class TopicRequest(BaseModel):
    topic: str

class OrchestrationResponse(BaseModel):
    topic: str
    result: str

@app.post("/generate-article", response_model=OrchestrationResponse)
def generate_article(request: TopicRequest):
    # fast api endpoint jo ki topic lega aur run_research_crew function ko call karega   
    try:
        result = run_research_crew(request.topic)
        return OrchestrationResponse(topic=request.topic, result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)