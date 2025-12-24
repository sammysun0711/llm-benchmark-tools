import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer  
from llmcompressor import oneshot  
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
 
  
# Patch get_input_embeddings function
def patch_get_input_embeddings():  
    def get_input_embeddings(self):  
        return self.thinker.get_input_embeddings()  
      
    # Overload method
    Qwen3OmniMoeForConditionalGeneration.get_input_embeddings = get_input_embeddings  

# Model Path
MODEL_ID = "/models/Qwen3-Omni-30B-A3B-Instruct"  
OUTPUT_DIR = "Qwen3-Omni-30B-A3B-Instruct-FP8-Dynamic-1224"
  
# Loda model and tokenizer
print("Loading model and tokenizer...")  

# Patch input_embedding before load model
patch_get_input_embeddings()  
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
    trust_remote_code=True 
)

# Load tokenizer
tokenizer = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)  
  
# Define FP8 Quantization config
# Specical configuration for Qwen3-Omni MOE model
recipe = QuantizationModifier(  
    targets="Linear",  
    scheme="FP8_DYNAMIC",  # W8A8 FP8 Dynamic Quantization
    ignore=[  
        "re:.*lm_head.*",                # LLM lm_head
        "model.embed_tokens",            # LLM input_embeddings
        "re:.*visual.*",                 # Vision encoder module
        "re:.*audio_tower.*",            # Audio encoder module
        #"re:multi_modal_projector.*",   # Multimodal projector (not avliable in Qwen3-Omni)
        "re:.*mlp.gate$",                # MoE expert gate
        #"re:.*mlp.shared_expert_gate$", # shared expert gate (only avaliable in talker)
    ]  
)  
  
# Ensure that model has get_input_embeddings method
assert hasattr(model, 'get_input_embeddings')  
print(model.get_input_embeddings())

# Apply FP8 PTPC quantization
print("Starting FP8 quantization...")  
oneshot(  
    model=model,
    tokenizer=tokenizer,
    dataset="open_platypus",          # Use calibration datasets
    recipe=recipe,  
    output_dir=OUTPUT_DIR,  
    max_seq_length=2048,              # Max seq length for calibration dataset, no change fof model sequence length
    num_calibration_samples=512,  
    #moe_calibrate_all_experts=True,  # Important, enable alll expert moe calibration
)  
  
print(f"Quantized model saved to {OUTPUT_DIR}")
 
