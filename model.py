# model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import get_peft_model, PromptTuningConfig, TaskType, PeftModel
from config import Config

class PEFTPromptTuningModel:
    def __init__(self, config: Config):
        self.config = config

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "right"  # causal LM usually pads right
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True  # Qwen models require this
        )

        # Configure PEFT for causal LM
        self.peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=config.prompt_tuning_init,
            num_virtual_tokens=config.num_virtual_tokens,
            prompt_tuning_init_text=config.prompt_tuning_init_text,
            tokenizer_name_or_path=config.model_name,
        )

        self.model = get_peft_model(self.base_model, self.peft_config)

        self.model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)

    @classmethod
    def load_pretrained(cls, config: Config, adapter_path: str):
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        instance = cls.__new__(cls)
        instance.config = config
        instance.tokenizer = tokenizer
        instance.base_model = base_model
        instance.model = model
        return instance

    def generate(self, input_ids, attention_mask, **kwargs):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            no_repeat_ngram_size=3,
            early_stopping=True,
            **kwargs
        )

    def get_data_collator(self):
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )

    def to(self, device):
        self.model = self.model.to(device)
        return self
