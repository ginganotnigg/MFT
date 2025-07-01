import torch
from model import PEFTPromptTuningModel
from data import get_question_type_instructions
from config import Config
import re

def parse_question_list_output(text, max_questions=None):
    text = re.sub(r'Example Response:.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    pattern = r'^\s*(\d+)\.\s*(.+?)(?=\n\s*\d+\.|\n\s*$|\Z)'
    matches = re.findall(pattern, text, flags=re.MULTILINE | re.DOTALL)

    questions = []
    for num_str, question in matches:
        if max_questions is not None and len(questions) >= max_questions:
            break

        question_lines = question.strip().split('\n')
        filtered_lines = [line for line in question_lines if not re.match(r'^\s*[A-D]\d?\.', line)]
        clean_question = ' '.join(filtered_lines).strip()

        q_mark_pos = clean_question.find('?')
        if q_mark_pos != -1:
            clean_question = clean_question[:q_mark_pos + 1].strip()

        if clean_question:
            questions.append(clean_question)

    return questions


def generate_multitopic_question_list(test_spec, model=None, config=None):
    """
    test_spec: dict with keys:
        - language: str
        - question_type: str
        - context: str
        - topics: list of dicts, each with:
            - topic: str
            - num_questions: int
            - difficulties: dict of {difficulty_level: count}
    """
    language = test_spec.get('language', 'English')
    question_type = test_spec.get('question_type', 'Multiple Choice')
    context = test_spec.get('context', '')
    topics = test_spec.get('topics', [])

    type_info = get_question_type_instructions(question_type)
    type_description = type_info['type_description']
    format_instruction = type_info['format_instruction']
    quality_guidelines = type_info['quality_guidelines']

    prompt_parts = []
    total_questions = 0
    combined_contexts = [context] if context else []

    for spec in topics:
        topic = spec['topic']
        num_q = spec['num_questions']
        difficulties = spec['difficulties']

        total_questions += num_q

        # Build difficulty breakdown string
        diff_str = ", ".join([f"{count} {level}" for level, count in difficulties.items()])

        prompt_parts.append(
            f"For {topic}, generate {num_q} {question_type.lower()} questions with difficulty distribution: {diff_str}."
        )

    combined_context = " ".join(combined_contexts) if combined_contexts else ""

    prompt = (
        "You are a technical educator generating a test with multiple topics and difficulty levels.\n"
        f"Instructions: Generate a total of {total_questions} questions divided as follows:\n"
    )

    for part in prompt_parts:
        prompt += f"{part}\n"

    if combined_context:
        prompt += f"\nContext:\n{combined_context}\n"

    prompt += (
        "Provide only a clean, numbered list of clear and concise questions.\n"
        "Do NOT include answer options or answers.\n"
        "Each question should have question mark. \n"
        f"Number questions sequentially from 1 to {total_questions}.\n"
        "Questions:\n1."
    )

    print(f"Generated prompt:\n{prompt}\n")

    inputs = model.tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_source_length
    )
    device = next(model.model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=config.max_new_tokens,
            num_beams=1,
            do_sample=True,
            temperature=getattr(config, "temperature", 0.8),
            top_p=getattr(config, "top_p", 0.9),
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
            early_stopping=True,
            repetition_penalty=1.1,
            length_penalty=1.0,
        )

    generated = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text:\n{generated}\n")

    del inputs, outputs
    torch.cuda.empty_cache()

    return parse_question_list_output(generated, max_questions=total_questions)


def main():
    config = Config()
    adapter_path = f"{config.output_dir}/final_model"
    model = PEFTPromptTuningModel.load_pretrained(config, adapter_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("Model loaded successfully!")

    test_spec = {
        "language": "English",
        "question_type": "Multiple Choice",
        "context": (
            "Databases store and organize data efficiently, supporting querying, indexing, and transactions. Databases include 2 types: SQL and NoSQL."
            "Data Structures and Algorithms cover arrays, linked lists, trees, graphs, sorting, searching, recursion and some advanced techniques like Dynamic Programming and Backtracking."
        ),
        "topics": [
            {
                "topic": "Databases",
                "num_questions": 5,
                "difficulties": {"Intern": 3, "Junior": 2}
            },
            {
                "topic": "Data Structures & Algorithms",
                "num_questions": 5,
                "difficulties": {"Junior": 3, "Mid": 2}
            }
        ]
    }

    questions = generate_multitopic_question_list(test_spec, model=model, config=config)

    print("Final generated questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

if __name__ == "__main__":
    main()