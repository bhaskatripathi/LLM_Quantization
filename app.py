import os
import subprocess
import streamlit as st
from huggingface_hub import snapshot_download, login, HfApi

if "quantized_model_path" not in st.session_state:
    st.session_state.quantized_model_path = None
if "upload_to_hf" not in st.session_state:
    st.session_state.upload_to_hf = False

def check_directory_path(directory_name: str) -> str:
    if os.path.exists(directory_name):
        path = os.path.abspath(directory_name)
        return str(path)

models_list = ['deepseek-ai/DeepSeek-R1', 'deepseek-ai/DeepSeek-V3', 
               'mistralai/Mistral-Small-24B-Instruct-2501', 'simplescaling/s1-32B', 
               'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 
               'deepseek-ai/DeepSeek-R1-Distill-Llama-70B', 'deepseek-ai/DeepSeek-R1-Zero', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 
               'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'm-a-p/YuE-s1-7B-anneal-en-cot', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B', 
               'microsoft/phi-4', 'huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated', 'meta-llama/Llama-3.3-70B-Instruct', 
               'cognitivecomputations/Dolphin3.0-R1-Mistral-24B', 'allenai/Llama-3.1-Tulu-3-405B', 'meta-llama/Llama-3.1-8B', 
               'meta-llama/Llama-3.1-8B-Instruct', 'Qwen/Qwen2.5-14B-Instruct-1M', 'mistralai/Mistral-Small-24B-Base-2501', 
               'huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated', 'huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2', 
               'Qwen/Qwen2.5-7B-Instruct-1M', 'open-thoughts/OpenThinker-7B', 'Almawave/Velvet-14B', 'cognitivecomputations/Dolphin3.0-Mistral-24B', 
               'Steelskull/L3.3-Damascus-R1', 'Qwen/Qwen2.5-Coder-32B-Instruct', 'huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated', 
               'cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese', 'jinaai/ReaderLM-v2', 'mistralai/Mistral-7B-Instruct-v0.3', 
               'meta-llama/Llama-3.2-1B', 'xwen-team/Xwen-7B-Chat', 'meta-llama/Llama-3.2-3B-Instruct', 'cognitivecomputations/DeepSeek-R1-AWQ', 
               'HuggingFaceTB/SmolLM2-1.7B-Instruct', 'xwen-team/Xwen-72B-Chat', 'openai-community/gpt2', 'meta-llama/Llama-2-7b-chat-hf', 'google/gemma-2-2b-it', 
               'mistralai/Mistral-7B-v0.1', 'meta-llama/Meta-Llama-3-8B', 'mistralai/Mistral-Nemo-Instruct-2407', 'microsoft/Phi-3.5-mini-instruct', 
               'arcee-ai/Virtuoso-Small-v2', 'MiniMaxAI/MiniMax-Text-01', 'AtlaAI/Selene-1-Mini-Llama-3.1-8B', 'Steelskull/L3.3-Nevoria-R1-70b', 
               'prithivMLmods/Calcium-Opus-14B-Elite2-R1', 'pfnet/plamo-2-1b', 'huihui-ai/DeepSeek-R1-Distill-Qwen-7B-abliterated-v2', 'Vikhrmodels/QVikhr-2.5-1.5B-Instruct-SMPO', 
               'mistralai/Mixtral-8x7B-Instruct-v0.1', 'vikhyatk/moondream2', 'meta-llama/Meta-Llama-3-8B-Instruct', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', 
               'Steelskull/L3.3-MS-Nevoria-70b', 'unsloth/DeepSeek-R1-Distill-Llama-8B', 'cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese', 'mistralai/Mistral-7B-Instruct-v0.2', 
               'deepseek-ai/DeepSeek-Coder-V2-Instruct', 'Qwen/Qwen2.5-32B', 'Qwen/Qwen2.5-72B-Instruct', 'allenai/Llama-3.1-Tulu-3-8B', 'SakanaAI/TinySwallow-1.5B-Instruct', 
               'm-a-p/YuE-s2-1B-general', 'arcee-ai/Virtuoso-Medium-v2', 'Black-Ink-Guild/Pernicious_Prophecy_70B', 'Qwen/Qwen2.5-14B', 'inflatebot/MN-12B-Mag-Mell-R1', 'Qwen/Qwen2.5-Math-1.5B', 
               'Qwen/Qwen2.5-Coder-7B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/QwQ-32B-Preview', 'NovaSky-AI/Sky-T1-32B-Preview', 'sometimesanotion/Lamarck-14B-v0.7', 
               'SentientAGI/Dobby-Mini-Leashed-Llama-3.1-8B', 'NaniDAO/deepseek-r1-qwen-2.5-32B-ablated', 'rubenroy/Zurich-14B-GCv2-5m', 'rubenroy/Geneva-12B-GCv2-5m', 'prithivMLmods/Primal-Opus-14B-Optimus-v1', 
               'prithivMLmods/Megatron-Opus-14B-Exp', 'prithivMLmods/Primal-Mini-3B-Exp', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'Qwen/Qwen2.5-0.5B-Instruct', 
               'Qwen/Qwen2.5-3B-Instruct', 'meta-llama/Llama-3.2-1B-Instruct', 'HuggingFaceTB/SmolLM2-135M-Instruct', 'PowerInfer/SmallThinker-3B-Preview', 
               'Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ', 'huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated', 'SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B', 
               'lightblue/DeepSeek-R1-Distill-Qwen-7B-Japanese', 'Ihor/Text2Graph-R1-Qwen2.5-0.5b', 'prithivMLmods/Bellatrix-Tiny-3B-R1', 'prithivMLmods/Bellatrix-Tiny-1.5B-R1', 'prithivMLmods/Megatron-Opus-14B-Stock', 
               'prithivMLmods/Jolt-v0.1', 'prithivMLmods/Sqweeks-7B-Instruct', 'bigscience/bloom', 'mistralai/Mistral-7B-Instruct-v0.1', 'google/gemma-2-27b-it', 'meta-llama/Llama-3.1-70B', 'Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2', 
               'Qwen/Qwen2.5-7B-Instruct', 'LatitudeGames/Wayfarer-12B', 'prithivMLmods/QwQ-Math-IO-500M', 'prithivMLmods/Llama-3.2-6B-AlgoCode', 'prithivMLmods/Omni-Reasoner-Merged', 'Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ', 
               'silma-ai/SILMA-Kashif-2B-Instruct-v1.0', 'mkurman/Qwen2.5-14B-DeepSeek-R1-1M', 'prithivMLmods/Blaze-14B-xElite', 'prithivMLmods/Megatron-Opus-7B-Exp', 'v2ray/GPT4chan-24B', 'prithivMLmods/Elita-1', 'prithivMLmods/Viper-Coder-v0.1', 
               'prithivMLmods/WebMind-7B-v0.1', 'prithivMLmods/Megatron-Corpus-14B-Exp.v2', 'prithivMLmods/Feynman-Grpo-Exp', 'meta-llama/Llama-2-7b-hf', 'microsoft/phi-2', 'Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF', 
               'google/gemma-2b', 'google/gemma-7b', 'sophosympatheia/Midnight-Miqu-70B-v1.5', 'jiviai/medX_v2', 'Alibaba-NLP/gte-Qwen2-7B-instruct', 'google/gemma-2-9b-it', 'meta-llama/Llama-Guard-3-8B', 'microsoft/Phi-3.5-vision-instruct', 
               'MarinaraSpaghetti/NemoMix-Unleashed-12B', 'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-32B-Instruct', 'meta-llama/Llama-3.2-3B', 'allenai/Molmo-7B-D-0924', 
               'HuggingFaceTB/SmolLM2-360M-Instruct', 'Zhengyi/LLaMA-Mesh', 'ibm-granite/granite-3.1-8b-instruct', 'livekit/turn-detector', 'SakanaAI/TinySwallow-1.5B', 'saheedniyi/YarnGPT', 
               'ContactDoctor/Bio-Medical-Llama-3-8B-CoT-012025', 'MiniMaxAI/MiniMax-VL-01', 'prithivMLmods/Omni-Reasoner4-Merged', 'unsloth/DeepSeek-R1', 'prithivMLmods/Calcium-Opus-14B-Elite2', 'prithivMLmods/Calcium-Opus-14B-Elite3', 
               'prithivMLmods/Bellatrix-Tiny-0.5B', 'prithivMLmods/Calcium-Opus-14B-Elite-Stock', 'prithivMLmods/Bellatrix-Tiny-1B', 'm-a-p/YuE-s1-7B-anneal-en-icl', 'arcee-ai/Virtuoso-Lite', 'stelterlab/Mistral-Small-24B-Instruct-2501-AWQ', 
               'prithivMLmods/Triangulum-v2-10B', 'prithivMLmods/Bellatrix-Tiny-1B-R1', 'huihui-ai/Mistral-Small-24B-Instruct-2501-abliterated', 'rubenroy/Gilgamesh-72B', 'rubenroy/Perseus-3192B', 'Nitral-Archive/NightWing3_Virtuoso-10B-v0.2', 
               'ibm-granite/granite-3.2-8b-instruct-preview', 'distilbert/distilgpt2', 'deepseek-ai/deepseek-coder-33b-instruct', 'microsoft/Phi-3-mini-4k-instruct', 'mistralai/Codestral-22B-v0.1', 'NovaSearch/stella_en_1.5B_v5', 'google/gemma-2-2b', 
               'lmms-lab/LLaVA-Video-7B-Qwen2', 'deepseek-ai/DeepSeek-V2.5', 'Qwen/Qwen2.5-Math-7B', 'AIDC-AI/Marco-o1', 'allenai/Llama-3.1-Tulu-3-8B-SFT', 'utter-project/EuroLLM-9B-Instruct', 'tiiuae/Falcon3-1B-Instruct', 
               'cognitivecomputations/DeepSeek-V3-AWQ', 'prithivMLmods/LwQ-10B-Instruct', 'prithivMLmods/LwQ-30B-Instruct', 'prithivMLmods/Calcium-20B', 'unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit', 
               'opensourcerelease/DeepSeek-R1-bf16', 'prithivMLmods/Llama-Express.1-Math', 'prithivMLmods/Llama-Express.1', 'prithivMLmods/Llama-Express.1-Tiny', 'prithivMLmods/Llama-Express.1-Merged', 
               'Delta-Vector/Rei-12B', 'kingabzpro/DeepSeek-R1-Medical-COT', 'prithivMLmods/Calme-Ties-78B', 'prithivMLmods/Qwen2.5-1.5B-DeepSeek-R1-Instruct', 'prithivMLmods/Calme-Ties2-78B', 'prithivMLmods/Bellatrix-Tiny-1B-v3', 
               'sometimesanotion/Qwenvergence-14B-v12-Prose-DS', 'TIGER-Lab/Qwen2.5-32B-Instruct-CFT', 'unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit', 'rubenroy/Geneva-12B-GCv2-1m', 'sometimesanotion/Qwenvergence-14B-v13-Prose-DS', 
               'deepseek-ai/deepseek-coder-6.7b-instruct', 'deepseek-ai/deepseek-moe-16b-base', 'deepseek-ai/deepseek-moe-16b-chat', 'microsoft/Phi-3-mini-128k-instruct', 'google/gemma-2-9b', 'AI-MO/NuminaMath-7B-TIR', 'CohereForAI/c4ai-command-r-plus-08-2024', 
               'Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24', 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF', 'CohereForAI/aya-expanse-8b', 'HuggingFaceTB/SmolLM2-135M', 'brgx53/3Blarenegv3-ECE-PRYMMAL-Martial', 'tiiuae/Falcon3-1B-Base', 
               'PocketDoc/Dans-PersonalityEngine-V1.1.0-12b', 'Kaoeiri/Magnum-v4-Cydonia-vXXX-22B', 'prithivMLmods/Blaze.1-32B-Instruct', 'kyutai/helium-1-preview-2b', 'prithivMLmods/Blaze.1-27B-Preview', 'prithivMLmods/Blaze.1-27B-Reflection', 
               'prithivMLmods/PyThagorean-10B', 'prithivMLmods/PyThagorean-3B', 'prithivMLmods/PyThagorean-Tiny', 'unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit', 'unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit', 'bespokelabs/Bespoke-Stratos-32B', 
               'Tarek07/Progenitor-V1.1-LLaMa-70B', 'mobiuslabsgmbh/DeepSeek-R1-ReDistill-Qwen-7B-v1.1', 'm-a-p/YuE-s1-7B-anneal-zh-cot', 'emredeveloper/DeepSeek-R1-Medical-COT', 'HelpingAI/HAI-SER', 'rubenroy/Geneva-12B-GCv2-10k', 'rubenroy/Geneva-12B-GCv2-50k', 
               'rubenroy/Geneva-12B-GCv2-100k', 'allura-org/GPT-J-6b-Disco-Elysium', 'fblgit/miniclaus-qw1.5B-UNAMGS-GRPO', 'suayptalha/Luminis-phi-4', 'EleutherAI/gpt-neo-2.7B', 'tiiuae/falcon-7b-instruct', 'deepseek-ai/deepseek-coder-1.3b-instruct', 
               'teknium/OpenHermes-2.5-Mistral-7B', 'maritaca-ai/sabia-7b', 'bigcode/starcoder2-3b', 'mistralai/Mixtral-8x7B-v0.1', 'Rijgersberg/GEITje-7B', 'segolilylabs/Lily-Cybersecurity-7B-v0.2', 'deepseek-ai/deepseek-coder-7b-instruct-v1.5', 
               'deepseek-ai/deepseek-math-7b-rl', 'SherlockAssistant/Mistral-7B-Instruct-Ukrainian', 'meta-llama/CodeLlama-7b-hf', 'databricks/dbrx-instruct', 'UnfilteredAI/Promt-generator', 'mistralai/Mixtral-8x22B-Instruct-v0.1', 'cognitivecomputations/dolphin-2.9-llama3-8b', 
               'ruslanmv/Medical-Llama3-8B', 'deepseek-ai/DeepSeek-V2-Chat', 'microsoft/llava-med-v1.5-mistral-7b', 'deepseek-ai/DeepSeek-V2-Lite-Chat', 'CohereForAI/aya-23-8B', 'ProbeMedicalYonseiMAILab/medllama3-v20', 'cognitivecomputations/dolphin-2.9.2-qwen2-72b', 
               'mlabonne/NeuralDaredevil-8B-abliterated', 'yentinglin/Llama-3-Taiwan-8B-Instruct', 'Sao10K/L3-8B-Stheno-v3.2', 'elyza/Llama-3-ELYZA-JP-8B', 'meta-llama/Llama-3.1-70B-Instruct', 'princeton-nlp/gemma-2-9b-it-SimPO', 'meta-llama/Llama-3.1-405B-Instruct', 
               'mistralai/Mistral-Nemo-Base-2407', 'unsloth/Meta-Llama-3.1-8B-Instruct', 'mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated', 'microsoft/maira-2', 'ContactDoctor/Bio-Medical-Llama-3-8B', 'ystemsrx/Qwen2-Boundless', 'upstage/solar-pro-preview-instruct', 
               'Epiculous/Violet_Twilight-v0.2', 'flowaicom/Flow-Judge-v0.1', 'Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-Math-1.5B-Instruct', 'meta-llama/Llama-Guard-3-1B', 'google/gemma-2-2b-jpn-it', 'unsloth/Llama-3.2-1B-Instruct', 'numind/NuExtract-1.5', 
               'rombodawg/Rombos-LLM-V2.5-Qwen-32b', 'anthracite-org/magnum-v4-22b', 'CohereForAI/aya-expanse-32b', 'VongolaChouko/Starcannon-Unleashed-12B-v1.0', 'Qwen/Qwen2.5-Coder-14B-Instruct', 'Qwen/Qwen2.5-Coder-32B', 'SmallDoge/Doge-60M', 'MaziyarPanahi/calme-3.2-instruct-78b', 
               'lianghsun/Llama-3.2-Taiwan-3B-Instruct', 'allenai/Llama-3.1-Tulu-3-8B-DPO', 'allenai/Llama-3.1-Tulu-3-70B', 'knifeayumu/Cydonia-v1.3-Magnum-v4-22B', 'utter-project/EuroLLM-9B', 'Skywork/Skywork-o1-Open-Llama-3.1-8B', 'Moraliane/SAINEMO-reMIX', 'LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct', 
               'NousResearch/Hermes-3-Llama-3.2-3B', 'recursal/QRWKV6-32B-Instruct-Preview-v0.1', 'allenai/OLMo-2-1124-13B-Instruct', 'huihui-ai/Llama-3.3-70B-Instruct-abliterated-finetuned-GPTQ-Int8', 'WiroAI/wiroai-turkish-llm-9b', 'SmallDoge/Doge-20M', 'FreedomIntelligence/HuatuoGPT-o1-70B', 
               'Sao10K/70B-L3.3-Cirrus-x1', 'internlm/internlm3-8b-instruct', 'prithivMLmods/PRM-Math-7B-Reasoner', 'prithivMLmods/QwQ-LCoT2-7B-Instruct', 'netease-youdao/Confucius-o1-14B', 'unsloth/DeepSeek-R1-Zero', 'unsloth/DeepSeek-R1-BF16', 'unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit',
               'suayptalha/Falcon3-Jessi-v0.4-7B-Slerp', 'RDson/CoderO1-DeepSeekR1-Coder-32B-Preview', 'bespokelabs/Bespoke-Stratos-7B', 'unsloth/DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit', 'RWKV-Red-Team/ARWKV-7B-Preview-0.1', 'lightblue/Karasu-DPO-7B', 'Spestly/Atlas-Pro-7B-Preview-1M', 
               'llm-jp/llm-jp-3-13b-instruct3', 'm-a-p/YuE-s1-7B-anneal-jp-kr-cot', 'm-a-p/YuE-s1-7B-anneal-jp-kr-icl', 'm-a-p/YuE-s1-7B-anneal-zh-icl', 'huihui-ai/Qwen2.5-14B-Instruct-1M-abliterated', 'AXCXEPT/phi-4-deepseek-R1K-RL-EZO', 'grimjim/DeepSauerHuatuoSkywork-R1-o1-Llama-3.1-8B',
               'sthenno/tempesthenno-icy-0130', 'neuralmagic/Mistral-Small-24B-Instruct-2501-FP8-Dynamic', 'Omartificial-Intelligence-Space/Arabic-DeepSeek-R1-Distill-8B', 'OddTheGreat/Badman_12B', 'MasterControlAIML/DeepSeek-R1-Strategy-Qwen-2.5-1.5b-Unstructured-To-Structured', 
               'rubenroy/Geneva-12B-GCv2-500k', 'bunnycore/Llama-3.2-3B-Bespoke-Thought', 'justinj92/Qwen2.5-1.5B-Thinking', 'RefalMachine/RuadaptQwen2.5-14B-Instruct', 'v2ray/GPT4chan-24B-QLoRA', 'CultriX/Qwen2.5-14B-Qwentangledv2', 'CultriX/Qwen2.5-14B-Ultimav2', 
               'Tarek07/Progenitor-V2.2-LLaMa-70B', 'dwetzel/DeepSeek-R1-Distill-Qwen-32B-GPTQ-INT4', 'Nitral-Archive/NightWing3-R1_Virtuoso-10B-v0.3e2', 'ucalyptus/prem-1B-grpo', 'Sakalti/Saka-14B', 'bunnycore/Qwen2.5-7B-MixStock-V0.1', 'braindao/DeepSeek-R1-Distill-Llama-8B-Uncensored', 
               'scb10x/llama3.1-typhoon2-deepseek-r1-70b', 'RefalMachine/RuadaptQwen2.5-14B-R1-distill-preview-v1', 
               'openai-community/gpt2-medium', 'openai-community/gpt2-xl', 'meta-llama/Llama-2-13b-hf', 'Trelis/Llama-2-7b-chat-hf-function-calling-v2', 'ByteWave/prompt-generator', 'HuggingFaceH4/zephyr-7b-beta', 'TheBloke/deepseek-llm-67b-chat-GPTQ', 'sarvamai/OpenHathi-7B-Hi-v0.1-Base', 
               'cognitivecomputations/dolphin-2.5-mixtral-8x7b', 
               'SanjiWatsuki/Sonya-7B', 'openchat/openchat-3.5-0106', 'ZySec-AI/SecurityLLM', 'defog/sqlcoder-70b-alpha', 'nakodanei/Blue-Orchid-2x7b', 'liuhaotian/llava-v1.6-mistral-7b', 'BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM', 'google/gemma-2b-it', 'bigcode/starcoder2-7b', 
               'nbeerbower/Maidphin-Kunoichi-7B-GGUF-Q4_K_M', 'HuggingFaceH4/starchat2-15b-v0.1', 'CohereForAI/c4ai-command-r-plus', 
               'HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1', 'UnfilteredAI/UNfilteredAI-1B', 'MaziyarPanahi/WizardLM-2-7B-GGUF', 'hiieu/Meta-Llama-3-8B-Instruct-function-calling-json-mode', 'shenzhi-wang/Llama3-8B-Chinese-Chat', 'Orenguteng/Llama-3-8B-Lexi-Uncensored', 'NTQAI/Nxcode-CQ-7B-orpo', 
               'lightblue/suzume-llama-3-8B-multilingual-orpo-borda-half', 'taide/Llama3-TAIDE-LX-8B-Chat-Alpha1', 'Nitral-AI/Poppy_Porpoise-0.72-L3-8B', 
               'WhiteRabbitNeo/Llama-3-WhiteRabbitNeo-8B-v2.0', 'marketeam/LLa-Marketing', 'microsoft/Phi-3-vision-128k-instruct', 'CohereForAI/aya-23-35B', 'shisa-ai/shisa-v1-llama3-8b', 'mistralai/Mistral-7B-v0.3', 'MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF', 'yentinglin/Llama-3-Taiwan-70B-Instruct', 
               'deepseek-ai/DeepSeek-Coder-V2-Lite-Base', 'Sao10K/L3-8B-Stheno-v3.3-32K', 'google/gemma-2-27b', 
               'Alibaba-NLP/gte-Qwen2-1.5B-instruct', 'm42-health/Llama3-Med42-8B', 'cognitivecomputations/dolphin-vision-7b', 'TheDrummer/Big-Tiger-Gemma-27B-v1', 'meta-llama/Llama-3.1-405B', 'google/shieldgemma-2b', 'amd/AMD-Llama-135m', 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit', 
               'aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored', 
               'NousResearch/Hermes-3-Llama-3.1-8B', 'mlabonne/TwinLlama-3.1-8B', 'ClosedCharacter/Peach-9B-8k-Roleplay', 'utter-project/EuroLLM-1.7B-Instruct', 'ai21labs/AI21-Jamba-1.5-Mini', 'Zyphra/Zamba2-2.7B-instruct', 'google/gemma-7b-aps-it', 'ifable/gemma-2-Ifable-9B', 'Qwen/Qwen2.5-1.5B', 
               'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-32B-Instruct-AWQ', 'brunopio/Llama3-8B-1.58-100B-tokens-GGUF', 'anthracite-org/magnum-v4-72b', 'nvidia/Llama-3_1-Nemotron-51B-Instruct', 'unsloth/Qwen2.5-14B-Instruct-bnb-4bit', 'katanemo/Arch-Function-3B', 'allenai/Molmo-7B-O-0924', 
               'unsloth/Llama-3.2-1B', 'lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct', 'BSC-LT/salamandra-2b-instruct', 'Steelskull/MSM-MS-Cydrion-22B', 'Bllossom/llama-3.2-Korean-Bllossom-3B', 'sam-paech/Delirium-v1', 'fblgit/TheBeagle-v2beta-32B-MGS', 'sarvamai/sarvam-1', 'HuggingFaceTB/SmolLM2-1.7B', 
               'Qwen/Qwen2.5-Coder-0.5B-Instruct', 'rombodawg/Rombos-Coder-V2.5-Qwen-14b', 'Nexusflow/Athene-V2-Chat', 'FallenMerick/MN-Violet-Lotus-12B', 'EVA-UNIT-01/EVA-Qwen2.5-72B-v0.2', 'allenai/OLMo-2-1124-7B-Instruct-preview', 'sometimesanotion/KytheraMix-7B-v0.2', 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct', 
               'LGAI-EXAONE/EXAONE-3.5-32B-Instruct', 'ibm-granite/granite-3.1-2b-instruct', 'unsloth/Llama-3.3-70B-Instruct-bnb-4bit', 'ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4', 'Sao10K/L3.3-70B-Euryale-v2.3', 'BSC-LT/ALIA-40b', 'huihui-ai/Llama-3.3-70B-Instruct-abliterated', 'SmallDoge/Doge-20M-Instruct', 
               'tiiuae/Falcon3-10B-Instruct', 'winninghealth/WiNGPT-Babel', 'FreedomIntelligence/HuatuoGPT-o1-8B', 'FreedomIntelligence/HuatuoGPT-o1-72B', 'prithivMLmods/Llama-3.1-5B-Instruct', 'prithivMLmods/Llama-Thinker-3B-Preview2', 'simplescaling/step-conditional-control-old', 'ngxson/MiniThinky-v2-1B-Llama-3.2', 
               'unsloth/phi-4-unsloth-bnb-4bit', 'KBlueLeaf/TIPO-500M-ft', 'bunnycore/Phi-4-RP-v0', 'Rombo-Org/Rombo-LLM-V2.5-Qwen-14b', 'nbeerbower/mistral-nemo-kartoffel-12B', 'sethuiyer/Llamaverse-3.1-8B-Instruct', 'Shaleen123/llama-3.1-8b-reasoning', 'Nohobby/L3.3-Prikol-70B-v0.3', 'nvidia/AceInstruct-1.5B', 'SmallDoge/Doge-20M-checkpoint', 
               'carsenk/llama3.2_1b_2025_uncensored_v2', 'bunnycore/Phi-4-Model-Stock-v2', 'Shaleen123/llama-3.1-8B-chain-reasoning', 'bunnycore/Phi-4-Model-Stock-v3', 'IVentureISB/MahaKumbh-Llama3.3-70B', 'DavidLanz/Llama-3.2-Taiwan-3B-Instruct', 'SmallDoge/Doge-60M-checkpoint', 'unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit', 
               'unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit', 'roleplaiapp/DeepSeek-R1-Distill-Qwen-32B-Q4_0-GGUF', 'arcee-ai/DeepSeek-R1-bf16', 'inarikami/DeepSeek-R1-Distill-Qwen-32B-AWQ', 'mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit', 'prithivMLmods/QwQ-LCoT1-Merged', 'prithivMLmods/Llama-3.2-3B-Math-Oct', 
               'Nitral-AI/Wayfarer_Eris_Noctis-12B', 'thirdeyeai/DeepSeek-R1-Distill-Qwen-7B-uncensored', 'NovaSky-AI/Sky-T1-32B-Flash', 'SZTAKI-HLT/Llama-3.2-1B-HuAMR', 'stepenZEN/DeepSeek-R1-Distill-Qwen-1.5B-Abliterated-dpo', 'mobiuslabsgmbh/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0', 'prithivMLmods/Phi-4-Super-1', 'prithivMLmods/Calcium-Opus-14B-Merge', 
               'prithivMLmods/COCO-7B-Instruct-1M', 'prithivMLmods/Taurus-Opus-7B', 'ReadyArt/L3.3-Nevoria-R1-70b_EXL2_5.0bpw_H8', 'NickyNicky/Llama-1B-GRPO_Final', 'unsloth/Qwen2.5-14B-Instruct-1M', 'RefalMachine/RuadaptQwen2.5-7B-Lite-Beta', 'hiieu/R1_tool_call_Distill-Qwen-7B', 'fla-hub/rwkv7-1.5B-world', 'KatyTheCutie/Repose-12B', 'DoppelReflEx/MN-12B-WolFrame', 
               'huihui-ai/DeepSeek-R1-Distill-Qwen-7B-abliterated', 'SubtleOne/Qwen2.5-32b-Erudite-Writer', 'ZeroXClem/Qwen2.5-7B-CelestialHarmony-1M', 'safe049/SmolTuring-8B-Instruct', 'unsloth/Mistral-Small-24B-Instruct-2501', 'unsloth/Mistral-Small-24B-Base-2501', 
               'llm-jp/llm-jp-3-150m-instruct3', 'llm-jp/llm-jp-3-7.2b-instruct3', 'suayptalha/Maestro-10B', 'Quazim0t0/Phi4.Turn.R1Distill_v1.5.1-Tensors', 'OddTheGreat/Malevolent_12B.v2', 'nbeerbower/Dumpling-Qwen2.5-7B-1k-r16', 'kromeurus/L3.1-Tivir-v0.1-10B', 'suayptalha/Maestro-R1-Llama-8B', 'rubenroy/Zurich-1.5B-GCv2-50k', 
               'rubenroy/Zurich-1.5B-GCv2-100k', 'rubenroy/Zurich-1.5B-GCv2-1m', 'enhanceaiteam/xea-llama', 'eridai/eridaRE', 'lianghsun/Marble-3B', 'DataSoul/MwM-7B-CoT-Merge1', 'Erland/Mistral-Small-24B-Base-ChatML-2501-bnb-4bit', 'chameleon-lizard/Qwen-2.5-7B-DTF', 'Vikhrmodels/QVikhr-2.5-1.5B-Instruct-SMPO_MLX-8bit', 'RecurvAI/Recurv-Clinical-Deepseek-R1', 
               'Darkhn/L3.3-Damascus-R1-5.0bpw-h8-exl2', 'Vikhrmodels/QVikhr-2.5-1.5B-Instruct-SMPO_MLX-4bit', 'BarBarickoza/Dans-Picaro-MagNante-v4-v1-12b-V3', 'skzxjus/Qwen2.5-7B-1m-Open-R1-Distill', 'CultriX/Qwen2.5-14B-Ultima', 'CultriX/Enhanced-TIES-Base-v1', 'loaiabdalslam/beetelware-saudi-R1-Distill-Llama-8B', 
               'Triangle104/Gemmadevi-Stock-10B', 'avemio-digital/German-RAG-HERMES-MOBIUS-R1-LLAMA', 'syubraj/MedicalChat-Phi-3.5-mini-instruct', 'Xiaojian9992024/Qwen2.5-THREADRIPPER-Small', 'jpacifico/Chocolatine-2-merged-qwen25arch', 'mobiuslabsgmbh/Meta-Llama-3-8B-Instruct_4bitgs64_hqq_hf', 'pabloce/esbieta-ec-qwen-2.5-3B', 'TareksLab/Progenitor-V2.3-LLaMa-70B', 
               'suayptalha/Lamarckvergence-14B', 'jpacifico/Chocolatine-2-14B-Instruct-v2.0.3', 'bunnycore/DeepThinker-7B-Sce-v2', 
               'sometimesanotion/Qwen2.5-7B-Gordion-v0.1', 'openai-community/gpt2-large', 'openai-community/openai-gpt', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-125m', 'GroNLP/gpt2-small-italian', 'LorenzoDeMattei/GePpeTto', 'Vamsi/T5_Paraphrase_Paws', 'ethzanalytics/distilgpt2-tiny-conversational', 'microsoft/DialoGPT-small', 'mrm8488/spanish-gpt2', 
               'shibing624/code-autocomplete-distilgpt2-python', 'EleutherAI/gpt-neox-20b', 'bigscience/bloom-560m', 'bigscience/bloom-1b7', 'rinna/japanese-gpt-neox-small', 'Langboat/bloom-1b4-zh',
               'EleutherAI/polyglot-ko-1.3b', 'bigscience/bloomz', 'Gustavosta/MagicPrompt-Stable-Diffusion', 'EleutherAI/polyglot-ko-5.8b', 'bigscience/bloomz-560m', 'bigscience/bloomz-3b', 'Norod78/gpt-fluentui-flat-svg', 'EleutherAI/pythia-160m', 'EleutherAI/pythia-1b-deduped', 'EleutherAI/pythia-12b', 'medalpaca/medalpaca-7b', 'huggyllama/llama-7b', 
               'vicgalle/gpt2-open-instruct-v1', 'bigcode/starcoder', 'TheBloke/stable-vicuna-13B-GPTQ', 'TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ', 'bigcode/tiny_starcoder_py', 'TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ', 'Monero/WizardLM-30B-Uncensored-Guanaco-SuperCOT-30b', 'TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ', 'nomic-ai/gpt4all-falcon', 
               'TheBloke/Karen_theEditor_13B-GPTQ', 'TheBloke/Nous-Hermes-13B-GPTQ', 'pankajmathur/orca_alpaca_3b', 'pankajmathur/orca_mini_3b', 'TheBloke/WizardLM-13B-V1-0-Uncensored-SuperHOT-8K-GPTQ', 'TheBloke/Wizard-Vicuna-13B-Uncensored-SuperHOT-8K-GPTQ', 
               'bigcode/starcoderbase-1b', 'NumbersStation/nsql-6B', 'HuggingFaceM4/idefics-80b', 'TheBloke/Pygmalion-7B-SuperHOT-8K-GPTQ', 'Maykeye/TinyLLama-v0', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-70b-chat-hf', 'TheBloke/Llama-2-13B-chat-GPTQ', 'NousResearch/Llama-2-7b-chat-hf', 'TheBloke/Llama-2-70B-Chat-GPTQ', 'NousResearch/Llama-2-13b-chat-hf', 'georgesung/llama2_7b_chat_uncensored', 'NousResearch/Nous-Hermes-Llama2-13b', 'TheBloke/30B-Epsilon-GPTQ', 'TheBloke/Dolphin-Llama-13B-GPTQ', 'bigcode/octocoder', 'Qwen/Qwen-7B', 'Qwen/Qwen-7B-Chat', 'uoe-nlp/gpt-neo-125m_instruction-tuned_sni', 'TheBloke/MythoMax-L2-13B-GPTQ', 'quantumaikr/llama-2-70b-fb16-korean', 'cenkersisman/gpt2-turkish-900m', 'codellama/CodeLlama-7b-hf', 'codellama/CodeLlama-13b-hf', 'codellama/CodeLlama-13b-Python-hf', 'codellama/CodeLlama-7b-Instruct-hf', 'codellama/CodeLlama-13b-Instruct-hf', 'codellama/CodeLlama-34b-hf', 'codellama/CodeLlama-34b-Python-hf', 'codellama/CodeLlama-34b-Instruct-hf', 'tiiuae/falcon-180B', 'uukuguy/speechless-llama2-luban-orca-platypus-13b', 'TinyLlama/TinyLlama-1.1B-step-50K-105b', 'diabolic6045/itineraries_Generator', '42dot/42dot_LLM-PLM-1.3B', '42dot/42dot_LLM-SFT-1.3B', 'tiiuae/falcon-180B-chat', 'PygmalionAI/pygmalion-2-13b', 'PygmalionAI/mythalion-13b', 'microsoft/phi-1_5', 'microsoft/phi-1', 'Undi95/UndiMix-v4-13B', 'teknium/Phi-Hermes-1.3B', 'TinyLlama/TinyLlama-1.1B-Chat-v0.1', 'AdaptLLM/medicine-LLM', 'AdaptLLM/law-LLM', 'AdaptLLM/finance-LLM', 'Dans-DiscountModels/Dans-RetroRodeo-13b', 'TheBloke/30B-Epsilon-AWQ', 'TheBloke/Wizard-Vicuna-7B-Uncensored-AWQ', 'TheBloke/Xwin-LM-13B-V0.1-GPTQ', 'Duxiaoman-DI/XuanYuan-70B', 'TheBloke/storytime-13B-GPTQ', 'Qwen/Qwen-14B-Chat', 'TheBloke/Mistral-7B-v0.1-AWQ', 'TheBloke/Mistral-7B-Instruct-v0.1-AWQ', 'TheBloke/Mistral-7B-v0.1-GPTQ', 'stabilityai/stablelm-3b-4e1t', 'rmanluo/RoG', 'lizpreciatior/lzlv_70b_fp16_hf', 'Dans-Archive/Dans-TotSirocco-7b', 'basilepp19/bloom-1b7_it', 'WisdomShell/CodeShell-7B', 'mychen76/mistral7b_ocr_to_json_v1', 'TheBloke/Athena-v4-GPTQ', 'HuggingFaceH4/zephyr-7b-alpha', 'cognitivecomputations/dolphin-2.1-mistral-7b', 'TheBloke/llava-v1.5-13B-AWQ', 'TheBloke/llava-v1.5-13B-GPTQ', 'THUDM/agentlm-7b', 'LumiOpen/Poro-34B', 'jondurbin/airoboros-m-7b-3.1.2', 'KoboldAI/LLaMA2-13B-Tiefighter-GPTQ', 'deepseek-ai/deepseek-coder-6.7b-base', 'aisingapore/sea-lion-3b', 'TRAC-MTRY/traclm-v1-3b-base', 'pfnet/plamo-13b-instruct', 'bkai-foundation-models/vietnamese-llama2-7b-40GB', 'flozi00/Mistral-7B-german-assistant-v4', 'TheBloke/zephyr-7B-beta-GPTQ', 'squarelike/Gugugo-koen-7B-V1.1', 'deepseek-ai/deepseek-coder-33b-base', 'TheBloke/Athnete-13B-GPTQ', 'TheBloke/Nethena-20B-GPTQ', 'cognitivecomputations/dolphin-2.2.1-mistral-7b', '01-ai/Yi-34B', 'TheBloke/deepseek-coder-33B-instruct-AWQ', 'alpindale/goliath-120b', 'Pclanglais/MonadGPT', 'epfl-llm/meditron-70b', 'epfl-llm/meditron-7b', 'alignment-handbook/zephyr-7b-sft-full', 'OpenLLM-France/Claire-7B-0.1', 'hakurei/mommygpt-3B', 'allenai/tulu-2-dpo-70b', 'NeverSleep/Noromaid-13b-v0.1.1', 'KoboldAI/LLaMA2-13B-Psyfighter2', 'Intel/neural-chat-7b-v3-1', 'OrionStarAI/OrionStar-Yi-34B-Chat', 'FPHam/Karen_TheEditor_V2_STRICT_Mistral_7B', 'Doctor-Shotgun/Nous-Capybara-limarpv3-34B', 'TinyLlama/TinyLlama-1.1B-Chat-v0.4', 'MohamedRashad/AceGPT-13B-chat-AWQ', 'THUDM/cogvlm-chat-hf', 'TheBloke/merlyn-education-safety-GPTQ', 'AntibodyGeneration/fine-tuned-progen2-small', 'TinyLlama/TinyLlama-1.1B-Chat-v0.6', 'OrionStarAI/OrionStar-Yi-34B-Chat-Llama', 'stabilityai/stablelm-zephyr-3b', 'FPHam/Karen_TheEditor_V2_CREATIVE_Mistral_7B', 'Jiayi-Pan/Tiny-Vicuna-1B', 'ethz-spylab/poisoned-rlhf-7b-SUDO-10', 'maywell/PiVoT-0.1-early', 'berkeley-nest/Starling-LM-7B-alpha', 'google/madlad400-8b-lm', 'SparseLLM/ReluLLaMA-7B', 'shleeeee/mistral-7b-wiki', 'ceadar-ie/FinanceConnect-13B', 'brucethemoose/CapyTessBorosYi-34B-200K-DARE-Ties-exl2-4bpw-fiction', 'TheBloke/saiga_mistral_7b-GPTQ', 'unsloth/llama-2-7b-bnb-4bit', 'Qwen/Qwen-72B-Chat', 'mlabonne/NeuralHermes-2.5-Mistral-7B', 'TheBloke/open-llama-3b-v2-wizard-evol-instuct-v2-196k-AWQ', 'TheBloke/deepseek-llm-7B-chat-GPTQ', 'beomi/Yi-Ko-6B', 'm-a-p/ChatMusician', 'maywell/Synatra-42dot-1.3B', 'Qwen/Qwen-Audio', 'Qwen/Qwen-Audio-Chat', 'mhenrichsen/context-aware-splitter-1b-english', 'jondurbin/cinematika-7b-v0.1', 'eci-io/climategpt-7b', 'simonveitner/MathHermes-2.5-Mistral-7B', 'ise-uiuc/Magicoder-DS-6.7B', 'ise-uiuc/Magicoder-S-DS-6.7B', 'migueldeguzmandev/paperclippetertodd3', 'sophosympatheia/Rogue-Rose-103b-v0.2', 'timpal0l/Mistral-7B-v0.1-flashback-v2', 'Trelis/Llama-2-7b-chat-hf-function-calling-v3', 'togethercomputer/StripedHyena-Nous-7B', 'Trelis/deepseek-llm-67b-chat-function-calling-v3', 'meta-llama/LlamaGuard-7b', 'openaccess-ai-collective/DPOpenHermes-7B-v2', 'tokyotech-llm/Swallow-7b-instruct-hf', 'AdaptLLM/finance-chat', 'AdaptLLM/law-chat', 'Intel/neural-chat-7b-v3-3', 'Rijgersberg/GEITje-7B-chat', 'TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T', 'TheBloke/Mistral-7B-Instruct-v0.2-AWQ', 'DaizeDong/GraphsGPT-2W', 'upstage/SOLAR-10.7B-Instruct-v1.0', 'upstage/SOLAR-10.7B-v1.0', 'w4r10ck/SOLAR-10.7B-Instruct-v1.0-uncensored', 'seyabde/mistral_7b_yo_instruct', 'TheBloke/dolphin-2.5-mixtral-8x7b-GPTQ', 'joey00072/ToxicHermes-2.5-Mistral-7B', 'THUDM/cogagent-vqa-hf', 'Rijgersberg/GEITje-7B-chat-v2', 'silk-road/ChatHaruhi_RolePlaying_qwen_7b', 'AdaptLLM/finance-LLM-13B', 'bkai-foundation-models/vietnamese-llama2-7b-120GB', 'scb10x/typhoon-7b', 'Felladrin/Llama-160M-Chat-v1', 'SuperAGI/SAM', 'Nero10578/Mistral-7B-Sunda-v1.0', 'NousResearch/Nous-Hermes-2-Yi-34B', 'ericpolewski/AIRIC-The-Mistral', 'charent/Phi2-Chinese-0.2B', 'unum-cloud/uform-gen', 'unsloth/mistral-7b-bnb-4bit', 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT', 'LR-AI-Labs/vbd-llama2-7B-50b-chat', 'unsloth/codellama-34b-bnb-4bit', 'cognitivecomputations/dolphin-2.6-mistral-7b', 'unsloth/llama-2-13b-bnb-4bit', 'OpenPipe/mistral-ft-optimized-1227', 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', 'GRMenon/mental-health-mistral-7b-instructv0.2-finetuned-V2', 'sethuiyer/SynthIQ-7b', 'unsloth/zephyr-sft-bnb-4bit', 'jondurbin/bagel-34b-v0.2', 'SkunkworksAI/tinyfrank-1.4B', 'NeuralNovel/Panda-7B-v0.1', 'unsloth/tinyllama-bnb-4bit', 'NousResearch/Nous-Hermes-2-SOLAR-10.7B', 'cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser', 'Vikhrmodels/Vikhr-7b-0.1', 'nicholasKluge/TeenyTinyLlama-460m', 'jsfs11/OH-dpov2', 'Unbabel/TowerBase-7B-v0.1', 'Doctor-Shotgun/Mixtral-8x7B-Instruct-v0.1-LimaRP-ZLoss', 'WizardLMTeam/WizardCoder-33B-V1.1', 'SanjiWatsuki/Kunoichi-7B', 'Unbabel/TowerInstruct-7B-v0.1', 'WYNN747/Burmese-GPT', 'NousResearch/Genstruct-7B', 'broskicodes/simple-stories-4M', 'STEM-AI-mtl/phi-2-electrical-engineering', 'mlabonne/phixtral-2x2_8', 'ross-dev/sexyGPT-Uncensored', 'HuggingFaceM4/VLM_WebSight_finetuned', 'stabilityai/stable-code-3b', 'huskyhong/noname-ai-v2_2-light', 'aari1995/germeo-7b-laser', 'argilla/distilabeled-OpenHermes-2.5-Mistral-7B', 'fblgit/UNA-TheBeagle-7b-v1', 'cognitivecomputations/MegaDolphin-120b', 'herisan/tinyllama-mental_health_counseling_conversations', 'NeverSleep/Noromaid-7B-0.4-DPO', 'therealcyberlord/TinyLlama-1.1B-Medical', 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO', 'szymonrucinski/Curie-7B-v1', 'MaziyarPanahi/Synatra-7B-v0.3-RP-Mistral-7B-Instruct-v0.2-slerp', 'SicariusSicariiStuff/Tenebra_30B_Alpha01_FP16', 'charlesdedampierre/TopicNeuralHermes-2.5-Mistral-7B', 'CodeGPTPlus/deepseek-coder-1.3b-typescript', 'herisan/Mistral-7b-bnb-4bit_mental_health_counseling_conversations', 'Viet-Mistral/Vistral-7B-Chat', 'sophosympatheia/Midnight-Rose-70B-v1.0', 'itsskofficial/falcon-7b-blooms-taxonomy-merged', 'AI-B/UTENA-7B-NSFW-V2', 'KoboldAI/LLaMA2-13B-Estopia', 'DiscoResearch/DiscoLM_German_7b_v1', 'CallComply/zephyr-7b-beta-32k', 'mlabonne/NeuralBeagle14-7B', 'jat-project/jat', 'macadeliccc/piccolo-math-2x7b', 'Isotonic/Dolphin-5.1-7b', 'shadowml/DareBeagle-7B', 'Karko/Proctora', 'haoranxu/ALMA-13B-R', 'yanolja/KoSOLAR-10.7B-v0.2', 'Tensoic/Kan-Llama-7B-SFT-v0.5', 'stabilityai/stablelm-2-1_6b', 'stabilityai/stablelm-2-zephyr-1_6b', 'lrds-code/boana-7b-instruct', 'vikhyatk/moondream1', 'gate369/Blurdus-7b-v0.1', 'Blizado/discolm-mfto-7b-german-v0.1', 'unsloth/mistral-7b-instruct-v0.2-bnb-4bit', 'senseable/WestLake-7B-v2', 'Qwen/Qwen1.5-0.5B', 'Qwen/Qwen1.5-1.8B', 'Qwen/Qwen1.5-7B', 'epinnock/deepseek-coder-33B-evol-feedback-v3', 'LanguageBind/MoE-LLaVA-StableLM-1.6B-4e', 'AISimplyExplained/Vakil-7B', 'RaviNaik/Llava-Phi2', 'motherduckdb/DuckDB-NSQL-7B-v0.1', 'deepseek-ai/deepseek-coder-7b-base-v1.5', 'KatyTheCutie/EstopianMaid-13B', 'abacusai/TheProfessor-155b', 'allenai/OLMo-1B', 'cfahlgren1/natural-functions', 'macadeliccc/WestLake-7B-v2-laser-truthy-dpo', 'jsfs11/WestOrcaDPO-7B-GTA', 'cckevinn/SeeClick', 'Unbabel/TowerInstruct-13B-v0.1', 'codellama/CodeLlama-70b-hf', 'codellama/CodeLlama-70b-Python-hf', 'codellama/CodeLlama-70b-Instruct-hf', 'seedboxai/KafkaLM-70B-German-V0.1', 'Qwen/Qwen1.5-7B-Chat', 'Qwen/Qwen1.5-72B-Chat', 'liuhaotian/llava-v1.6-vicuna-7b', 'liuhaotian/llava-v1.6-vicuna-13b', 'LoneStriker/Lily-Cybersecurity-7B-v0.2-8.0bpw-h8-exl2', 'Qwen/Qwen1.5-0.5B-Chat', 'unsloth/codellama-7b-bnb-4bit', 'Gille/StrangeMerges_17-7B-dare_ties', 'Gille/StrangeMerges_19-7B-dare_ties', 'Bread-AI/Crumb-13B', 'Druvith/MEDMISTRAL', 'alchemonaut/BoreanGale-70B', 'Gille/StrangeMerges_20-7B-slerp', 'PipableAI/pip-SQL-1B', 'cais/HarmBench-Llama-2-13b-cls', 'sophosympatheia/Midnight-Rose-70B-v2.0.3', 'defog/sqlcoder-7b-2', 'RUCKBReasoning/TableLLM-13b', 'RUCKBReasoning/TableLLM-7b', 'Sao10K/Fimbulvetr-11B-v2', 'nvidia/OpenMath-Mistral-7B-v0.1-hf', 'yanolja/EEVE-Korean-10.8B-v1.0', 'WhiteRabbitNeo/Trinity-33B-v1.0', 'hon9kon9ize/CantoneseLLM-6B-preview202402', 'Nitral-Archive/Pasta-Lake-7b', 'kennylam/Breeze-7B-Cantonese-v0.1', 'Unbabel/TowerInstruct-7B-v0.2', 'GritLM/GritLM-7B', 'google/gemma-7b-it', 'ytu-ce-cosmos/turkish-gpt2-large', 'prometheus-eval/prometheus-7b-v2.0', 'NingLab/eCeLLM-M', 'PipableAI/pip-sql-1.3b', 'rhplus0831/maid-yuzu-v8', 'proxectonos/Carballo-bloom-1.3B', 'sambanovasystems/SambaLingo-Arabic-Chat', 'shahzebnaveed/StarlingHermes-2.5-Mistral-7B-slerp', 'LumiOpen/Viking-7B', 'tanamettpk/TC-instruct-DPO', 'Tann-dev/sex-chat-dirty-girlfriend', 'BioMistral/BioMistral-7B-DARE-AWQ-QGS128-W4-GEMM', 'NousResearch/Nous-Hermes-2-Mistral-7B-DPO', 'SparseLLM/prosparse-llama-2-7b', 'HuggingFaceTB/cosmo-1b', 'Efficient-Large-Model/VILA-13b', 'scb10x/typhoon-7b-instruct-02-19-2024', 'LumiOpen/Viking-33B', 'prometheus-eval/prometheus-8x7b-v2.0', 'bigcode/starcoder2-15b', 'togethercomputer/evo-1-131k-base', 'unsloth/gemma-7b-bnb-4bit', 'unsloth/gemma-2b-bnb-4bit', 'unsloth/gemma-2b-it-bnb-4bit', 'unsloth/gemma-7b-it-bnb-4bit', 'yanolja/EEVE-Korean-Instruct-10.8B-v1.0', 'yanolja/EEVE-Korean-2.8B-v1.0', 'yanolja/EEVE-Korean-Instruct-2.8B-v1.0', 'gordicaleksa/YugoGPT', 'timpal0l/Mistral-7B-v0.1-flashback-v2-instruct', 'allenai/OLMo-7B-Instruct', 'coggpt/qwen-1.5-patent-translation', 'GreatCaptainNemo/ProLLaMA', 'Felladrin/Minueza-32M-Base', 'Felladrin/Minueza-32M-Chat', 'm-a-p/OpenCodeInterpreter-DS-1.3B', 'MaziyarPanahi/LongAlpaca-13B-GGUF', 'OPI-PG/Qra-1b', 'MathGenie/MathGenie-InterLM-20B', 'MaziyarPanahi/Mistral-7B-Instruct-Aya-101', 'ENERGY-DRINK-LOVE/eeve_dpo-v3', 'Stopwolf/Tito-7B-slerp', 'MaziyarPanahi/Mistral-7B-Instruct-Aya-101-GGUF', 'PORTULAN/gervasio-7b-portuguese-ptbr-decoder', 'JinghuiLuAstronaut/DocLLM_baichuan2_7b', 'vicgalle/RoleBeagle-11B', 'HuggingFaceH4/zephyr-7b-gemma-v0.1', 'KatyTheCutie/LemonadeRP-4.5.3', 'Kooten/LemonadeRP-4.5.3-4bpw-exl2', 'sophosympatheia/Midnight-Miqu-103B-v1.0', 'soketlabs/pragna-1b', 'remyxai/SpaceLLaVA', 'Efficient-Large-Model/VILA-2.7b', 'hiyouga/Llama-2-70b-AQLM-2Bit-QLoRA-function-calling', 'occiglot/occiglot-7b-de-en-instruct', 'erythropygia/Gemma2b-Turkish-Instruction', 'state-spaces/mamba-2.8b-hf', 'state-spaces/mamba-130m-hf', 'zamal/gemma-7b-finetuned', 'Divyanshu04/LLM3', 'yam-peleg/Hebrew-Gemma-11B', 'yam-peleg/Hebrew-Gemma-11B-Instruct', 'stabilityai/stable-code-instruct-3b', 'Gille/StrangeMerges_35-7B-slerp', 'stanford-oval/llama-7b-wikiwebquestions', 'cstr/Spaetzle-v8-7b', 'ChaoticNeutrals/BuRP_7B', 'cstr/Spaetzle-v12-7b', 'lightblue/ao-karasu-72B', 'NousResearch/Hermes-2-Pro-Mistral-7B', 'hiieu/Vistral-7B-Chat-function-calling', 'CohereForAI/c4ai-command-r-v01', 'ND911/Franken-Mistral-Merlinite-Maid', 'fhai50032/Mistral-4B', 'meta-llama/CodeLlama-7b-Python-hf', 'meta-llama/CodeLlama-7b-Instruct-hf', 'meta-llama/CodeLlama-13b-hf', 'meta-llama/CodeLlama-13b-Instruct-hf', 'ministral/Ministral-3b-instruct', 'CohereForAI/c4ai-command-r-v01-4bit', 'KissanAI/Dhenu-vision-lora-0.1', 'MaziyarPanahi/Calme-7B-Instruct-v0.2', 'icefog72/Kunokukulemonchini-7b-4.1bpw-exl2', 'ChaoticNeutrals/Infinitely-Laydiculous-7B', 'Virt-io/Nina-v2-7B', 'BAAI/bge-reranker-v2-minicpm-layerwise', 'NexaAIDev/Octopus-v2', 'jhu-clsp/FollowIR-7B', 'cais/HarmBench-Mistral-7b-val-cls', 'ezelikman/quietstar-8-ahead', 'szymonrucinski/Krakowiak-7B-v3', 'FluffyKaeloky/Midnight-Miqu-103B-v1.5', 'Nekochu/Confluence-Renegade-7B', 'fxmarty/tiny-dummy-qwen2', 'ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1', 'ChaoticNeutrals/Eris_PrimeV3-Vision-7B', 'somosnlp/Sam_Diagnostic', 'google/codegemma-2b', 'google/codegemma-7b', 'google/codegemma-7b-it', 'stabilityai/stablelm-2-12b', 'unsloth/mistral-7b-v0.2-bnb-4bit', 'Praneeth/code-gemma-2b-it', 'Inv/Konstanta-V4-Alpha-7B', 'liminerity/e.star.7.b', 'Sahi19/Gemma2bLegalChatbot', 'gokaygokay/moondream-prompt', 'YanweiLi/MGM-7B', 'beomi/gemma-ko-2b', 'Anant58/Genshin-chat-ARM', 'thtskaran/sanskritayam-gpt', 'Natkituwu/Erosumika-7B-v3-7.1bpw-exl2', 'MarsupialAI/SkunkApe-14b', 'google/gemma-1.1-7b-it', 'Smuggling1710/InfinToppyKuno-DARE-7b', 'botbot-ai/CabraQwen7b', 'bsen26/113-Aspect-Emotion-Model', 'arcee-ai/Saul-Nous-Hermes-2-Mistral-7B-DPO-Ties', 'cognitivecomputations/dolphin-2.8-mistral-7b-v02', 'ai21labs/Jamba-v0.1', 'grimjim/Mistral-Starling-merge-trial1-7B', 'mikewang/PVD-160k-Mistral-7b', 'Eurdem/Pinokio_v1.0', 'keeeeenw/MicroLlama', '1bitLLM/bitnet_b1_58-3B', '1bitLLM/bitnet_b1_58-xl', '1bitLLM/bitnet_b1_58-large', 'EdBerg/MISTRALNEURAL-7B-slerp', 'Kukedlc/Neural-4-QA-7b']

# Define quantization types
QUANT_TYPES = ["Q2_K", "Q3_K_l", "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_1", "Q4_K_M", "Q4_K_S", "Q5_0", "Q5_1", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "BF16", "F16", "F32"]

QUANT_DESCRIPTIONS = {
    "Q2_K": "Smallest size, acceptable for simple tasks",
    "Q3_K_l": "Good balance for lightweight applications",
    "Q3_K_M": "Medium quality, good for general text generation",
    "Q3_K_S": "Small size, suitable for simple interactions",
    "Q4_0": "Legacy format, basic compression",
    "Q4_1": "Better than Q4_0, good general purpose",
    "Q4_K_M": "Recommended for most uses, good balance",
    "Q4_K_S": "Smaller than Q4_K_M, still good quality",
    "Q5_0": "Higher precision than Q4, legacy format",
    "Q5_1": "Improved Q5, good for complex tasks",
    "Q5_K_M": "High quality, larger size, good for complex reasoning",
    "Q5_K_S": "Balanced quality and size in Q5 family",
    "Q6_K": "Very high quality, larger size",
    "Q8_0": "Highest quality quantized, largest size",
    "BF16": "Brain Float 16, good for GPU inference",
    "F16": "Full 16-bit precision, high accuracy",
    "F32": "Full 32-bit precision, highest accuracy, largest size"
}

model_dir_path = check_directory_path("/app/llama.cpp")

def download_model(hf_model_name, output_dir="/tmp/models"):
    """
    Downloads a Hugging Face model and saves it locally.
    """
    st.write(f"📥 Downloading `{hf_model_name}` from Hugging Face...")
    os.makedirs(output_dir, exist_ok=True)
    snapshot_download(repo_id=hf_model_name, local_dir=output_dir, local_dir_use_symlinks=False)
    st.success("✅ Model downloaded successfully!")

def convert_to_gguf(model_dir, output_file):
    """
    Converts a Hugging Face model to GGUF format.
    """
    st.write(f"🔄 Converting `{model_dir}` to GGUF format...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cmd = [
        "python3", "/app/llama.cpp/convert_hf_to_gguf.py", model_dir,
        "--outfile", output_file
    ]
    process = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode == 0:
        st.success(f"✅ Conversion complete: `{output_file}`")
    else:
        st.error(f"❌ Conversion failed: {process.stderr}")

def quantize_llama(model_path, quantized_output_path, quant_type):
    """
    Quantizes a GGUF model.
    """
    st.write(f"⚡ Quantizing `{model_path}` with `{quant_type}` precision...")
    os.makedirs(os.path.dirname(quantized_output_path), exist_ok=True)
    quantize_path = "/app/llama.cpp/build/bin/llama-quantize"
    
    cmd = [
        "/app/llama.cpp/build/bin/llama-quantize", 
        model_path, 
        quantized_output_path,
        quant_type
    ]
    
    process = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if process.returncode == 0:
        st.success(f"✅ Quantized model saved at `{quantized_output_path}`")
    else:
        st.error(f"❌ Quantization failed: {process.stderr}")

def automate_llama_quantization(hf_model_name, quant_type):
    """
    Orchestrates the entire quantization process.
    """
    output_dir = "/tmp/models"
    gguf_file = os.path.join(output_dir, f"{hf_model_name.replace('/', '_')}.gguf")
    quantized_file = gguf_file.replace(".gguf", f"-{quant_type}.gguf")

    progress_bar = st.progress(0)

    # Step 1: Download
    st.write("### Step 1: Downloading Model")
    download_model(hf_model_name, output_dir)
    progress_bar.progress(33)

    # Step 2: Convert to GGUF
    st.write("### Step 2: Converting Model to GGUF Format")
    convert_to_gguf(output_dir, gguf_file)
    progress_bar.progress(66)

    # Step 3: Quantize Model
    st.write("### Step 3: Quantizing Model")
    quantize_llama(gguf_file, quantized_file, quant_type.lower())
    progress_bar.progress(100)

    st.success(f"🎉 All steps completed! Quantized model available at: `{quantized_file}`")
    return quantized_file

def upload_to_huggingface(file_path, repo_id, token):
    """
    Uploads a file to Hugging Face Hub.
    """
    try:
        # Log in to Hugging Face
        login(token=token)

        # Initialize HfApi
        api = HfApi()

        # Create the repository if it doesn't exist
        api.create_repo(repo_id, exist_ok=True, repo_type="model")

        # Upload the file
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=repo_id,
        )
        st.success(f"✅ File uploaded to Hugging Face: {repo_id}")

        # Reset session state and rerun
        st.session_state.quantized_model_path = None
        st.session_state.upload_to_hf = False
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to upload file: {e}")

st.title("🦙 LLaMA Model Quantization (llama.cpp)")


selected_model = st.selectbox("Select the Hugging Face Model", models_list, index=None)
hf_model_name = selected_model if selected_model else st.text_input("Enter Hugging Face Model (If not there in the above list)")

selected_quant = st.selectbox(
    "Select Quantization Type",
    QUANT_TYPES,
    help="Hover over options to see descriptions",
    format_func=lambda x: f"{x} - {QUANT_DESCRIPTIONS[x]}"
)
start_button = st.button("🚀 Start Quantization")

if start_button:
    if hf_model_name and selected_quant:
        with st.spinner("Processing..."):
            st.session_state.quantized_model_path = automate_llama_quantization(hf_model_name, selected_quant)
    else:
        st.warning("Please select/enter the necessary fields.")

if st.session_state.quantized_model_path:
    with open(st.session_state.quantized_model_path, "rb") as f:
        if st.download_button("⬇️ Download Quantized Model", f, file_name=os.path.basename(st.session_state.quantized_model_path)):
            st.session_state.quantized_model_path = None
            st.session_state.upload_to_hf = False
            st.rerun()
    
    # Checkbox for upload section
    st.session_state.upload_to_hf = st.checkbox("Upload to Hugging Face", value=st.session_state.upload_to_hf)
    
    if st.session_state.upload_to_hf:
        st.write("### Upload to Hugging Face")
        repo_id = st.text_input("Enter Hugging Face Repository ID (e.g., 'username/repo-name')")
        hf_token = st.text_input("Enter Hugging Face Token", type="password")
        
        if st.button("📤 Upload to Hugging Face"):
            if repo_id and hf_token:
                with st.spinner("Uploading..."):
                    upload_to_huggingface(st.session_state.quantized_model_path, repo_id, hf_token)
            else:
                st.warning("Please provide a valid repository ID and Hugging Face token.")