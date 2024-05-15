from transformers import AutoTokenizer,BitsAndBytesConfig
from abc import ABC
from ctransformers import AutoModelForCausalLM
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun


device = "cpu" # the device to load the model onto
model_name = "Qwen/Qwen1.5-1.8B-Chat-GGUF"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    # torch_dtype="auto",
    # device_map="auto",
    # trust_remote_code=True,
    # cache_dir="./Qwen1.5-0.5B-Chat"
    # offload_folder="offload",
    hf = True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir="./Qwen1.5-0.5B-Chat")
# from optimum.bettertransformer import BetterTransformer
# model = BetterTransformer.transform(model)
class Qwen(LLM, ABC):
     max_token: int = 10000
     temperature: float = 0.01
     top_p = 0.9
     history_len: int = 3

     def __init__(self):
         super().__init__()

     @property
     def _llm_type(self) -> str:
         return "Qwen"

     @property
     def _history_len(self) -> int:
         return self.history_len

     def set_history_len(self, history_len: int = 10) -> None:
         self.history_len = history_len

     def _call(
         self,
         prompt: str,
         stop: Optional[List[str]] = None,
         run_manager: Optional[CallbackManagerForLLMRun] = None,
     ) -> str:
         messages = [
             {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": prompt}
         ]
         text = tokenizer.apply_chat_template(
             messages,
             tokenize=False,
             add_generation_prompt=True
         )
         model_inputs = tokenizer([text], return_tensors="pt").to(device)
         generated_ids = model.generate(
             model_inputs.input_ids,
             max_new_tokens=4096,
             pad_token_id=tokenizer.eos_token_id
         )
         generated_ids = [
             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
         ]

         response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
         return response

     @property
     def _identifying_params(self) -> Mapping[str, Any]:
         """Get the identifying parameters."""
         return {"max_token": self.max_token,
                 "temperature": self.temperature,
                 "top_p": self.top_p,
                 "history_len": self.history_len}