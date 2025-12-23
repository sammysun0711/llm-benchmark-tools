import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer  
from llmcompressor import oneshot  
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
 
  
# 永久修补类定义  
def patch_get_input_embeddings():  
    def get_input_embeddings(self):  
        return self.thinker.get_input_embeddings()  
      
    # 确保方法被正确添加到类上  
    Qwen3OmniMoeForConditionalGeneration.get_input_embeddings = get_input_embeddings  

# 模型标识  
MODEL_ID = "/models/Qwen3-Omni-30B-A3B-Instruct"  
OUTPUT_DIR = "/models/Qwen3-Omni-30B-A3B-Instruct-FP8-Dynamic"  
  
# 加载模型和分词器  
print("Loading model and tokenizer...")  

# 在加载模型前调用  
patch_get_input_embeddings()  
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
    trust_remote_code=True 
)
tokenizer = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)  
  
# 定义 FP8 量化方案  
# 针对多模态 MoE 模型的特殊配置  
recipe = QuantizationModifier(  
    targets="Linear",  
    scheme="FP8_DYNAMIC",  # W8A8 FP8 动态量化  
    ignore=[  
        "re:.*lm_head.*",             # 语言模型头部  
        "model.embed_tokens",         # 输入嵌入层  
        "re:.*visual.*",          # 视觉塔组件
        "re:.*audio_tower.*",  
        "re:multi_modal_projector.*", # 多模态投影层  
        "re:.*mlp.gate$",             # MoE 门控层  
        "re:.*mlp.shared_expert_gate$", # 共享专家门控  
    ]  
)  
  
# 验证方法存在  
assert hasattr(model, 'get_input_embeddings')  
print(model.get_input_embeddings())  # 应该成功  
# 执行量化  
print("Starting FP8 quantization...")  
oneshot(  
    model=model,
    tokenizer=tokenizer,
    dataset="open_platypus",  # 使用内置数据集进行校准  
    recipe=recipe,  
    output_dir=OUTPUT_DIR,  
    max_seq_length=2048,  
    num_calibration_samples=512,  
    #moe_calibrate_all_experts=True,  # 重要：启用所有专家校准  
)  
  
print(f"Quantized model saved to {OUTPUT_DIR}")
 
