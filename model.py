# model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import get_peft_model, PromptTuningConfig, TaskType, PeftConfig
from config import Config

class PEFTPromptTuningModel:
    def __init__(self, config: Config):
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True, local_files_only=True
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True
        )

        self.peft_config = PeftConfig.from_pretrained(
            config.adapter_path, trust_remote_code=True, local_files_only=True
        )

        self.model = get_peft_model(self.base_model, self.peft_config)
        self.model.print_trainable_parameters()

    @classmethod
    def load_pretrained(cls, config: Config):
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, local_files_only=True, trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=True
        )
        peft_config = PeftConfig.from_pretrained(
            config.adapter_path, trust_remote_code=True, local_files_only=True
        )
        model = get_peft_model(base_model, peft_config)
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
