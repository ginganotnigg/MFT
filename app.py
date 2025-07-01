from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import generate_multitopic_question_list
from typing import List, Optional
from model import PEFTPromptTuningModel
from config import Config
import torch
import uvicorn

class DifficultyDistribution(BaseModel):
    Intern: Optional[int] = 0
    Junior: Optional[int] = 0
    Middle: Optional[int] = 0
    Senior: Optional[int] = 0
    Lead: Optional[int] = 0

class Topic(BaseModel):
    name: str
    difficultyDistribution: DifficultyDistribution

class Context(BaseModel):
    text: str
    links: List[str] = []

class SuggestExamQuestionRequest(BaseModel):
    question_type: str
    language: str
    topics: List[Topic]
    creativity: Optional[int] = 5
    context: Optional[Context] = None

class SuggestExamQuestionResponse(BaseModel):
    questions: List[str]

app = FastAPI()

# Global model and device
model = None
device = None
config = Config()

@app.on_event("startup")
def load_model():
    global model, device
    adapter_path = f"{config.output_dir}/final_model"
    model = PEFTPromptTuningModel.load_pretrained(config, adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

@app.get("/health")
async def health():
    return {"status": "OK"}

@app.post("/generate", response_model=SuggestExamQuestionResponse)
async def generate_questions(request: SuggestExamQuestionRequest):
    topics = []
    for topic in request.topics:
        difficulties = {}
        num_questions = 0
        dd = topic.difficultyDistribution
        for level in ["Intern", "Junior", "Middle", "Senior", "Lead"]:
            val = getattr(dd, level, 0)
            if val:
                difficulties[level] = val
                num_questions += val
        topics.append({
            "topic": topic.name,
            "num_questions": num_questions,
            "difficulties": difficulties
        })

    test_spec = {
        "language": request.language,
        "question_type": request.question_type,
        "context": request.context.text if request.context else "",
        "topics": topics
    }

    questions = generate_multitopic_question_list(test_spec, model=model, config=config)

    del test_spec

    return SuggestExamQuestionResponse(questions=questions)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7777, log_level="info")