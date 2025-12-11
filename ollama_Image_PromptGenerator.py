#!/usr/bin/env python
# coding: utf-8

# ================================================================
# EXECUTIVE SUMMARY : Ollama + Stable Diffusion Prompt Generator
# ================================================================
# Purpose:
# - Automate the generation of high‑quality image prompts using Ollama LLM.
# - Feed enhanced prompts into Stable Diffusion WebUI for txt2img generation.
#
# Why it matters:
# - Recruiters and collaborators want to see end‑to‑end pipelines that combine
#   LLMs with generative models.
# - This script demonstrates environment checks, model auto‑discovery, API calls,
#   error handling, and image saving.
#
# Techniques highlighted:
# - Environment validation (PyTorch, CUDA, xFormers).
# - Model auto‑discovery with filesystem scanning.
# - REST API integration with Ollama and Stable Diffusion.
# - Prompt parsing into positive/negative components.
# - Image generation with configurable parameters.
# ================================================================


# ==== Step 1: Environment Check & Dependencies ====
# Why: Ensures PyTorch, TorchVision, and xFormers are installed and CUDA is available.
# Recruiter insight: Shows you validate GPU acceleration before running heavy models.
# Syntax notes:
# - torch.__version__ → returns installed PyTorch version.
# - torch.cuda.is_available() → boolean check for GPU support.
# - torch.cuda.get_device_name(0) → name of first CUDA device.

import torch
import torchvision
import xformers

print("🔍 Environment Check:")
print(f"  PyTorch Version: {torch.__version__}")
print(f"  TorchVision Version: {torchvision.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
print(f"  xFormers Version: {xformers.__version__}")


# ==== Step 2: Configuration & Utility Functions ====
# Purpose: Define API endpoints, available models, and helper functions.
# Why: Recruiters see automation (auto-discovery of models) and reproducibility.
# Technique: os.listdir() + filtering by extensions (.safetensors, .ckpt).

import requests
import base64
import json
from typing import Optional, Tuple, Dict, Any
from PIL import Image
from io import BytesIO
from IPython.display import display
import time
import os

# --- Configuration ---
SD_API_URL = "http://127.0.0.1:7860"              # Stable Diffusion WebUI
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama API

TIMEOUT_OLLAMA = 120 # Seconds for prompt generation
TIMEOUT_SD_API = 300 # Seconds for image generation

MODEL_FOLDER_PATH = r"C:\AI\stable-diffusion-webui-master\models\Stable-diffusion"
SUPPORTED_EXTENSIONS = ('.safetensors', '.ckpt')

def auto_discover_models(folder_path: str) -> list[str]:
    """Lists model files in the specified directory."""
    if not os.path.isdir(folder_path):
        print(f"❌ Error: Model directory not found at: {folder_path}")
        return []
    models = [f for f in os.listdir(folder_path) if f.endswith(SUPPORTED_EXTENSIONS)]
    models.sort()
    return models

AVAILABLE_MODELS = auto_discover_models(MODEL_FOLDER_PATH)

if not AVAILABLE_MODELS:
    print("⚠️ Warning: No models found via auto-discovery. Using a fallback list.")
    AVAILABLE_MODELS = ["v1-5-pruned-emaonly.safetensors"]
    MODEL_SELECTION_INDEX = 0
elif len(AVAILABLE_MODELS) > 1:
    MODEL_SELECTION_INDEX = 1
else:
    MODEL_SELECTION_INDEX = 0

MODEL_TO_USE = AVAILABLE_MODELS[MODEL_SELECTION_INDEX]

NEGATIVE_PROMPT_KEYWORDS = "low quality, blurry, worst quality, extra limbs, deformed, bad anatomy, jpeg artifacts"

print(f"✅ Configuration Loaded")
print(f"  Total Models Found: {len(AVAILABLE_MODELS)}")
print(f"  Selected Model Index ({MODEL_SELECTION_INDEX}): {MODEL_TO_USE}")
print(f"  SD API: {SD_API_URL}")
print(f"  Ollama API: {OLLAMA_URL}")


# ==== Utility Function: Switch Stable Diffusion Checkpoint ====
# Purpose: Calls the WebUI API to change the active model.
# Why: Demonstrates REST API integration and payload design.
# Syntax:
# - requests.post(endpoint, json=payload) → sends JSON to API.
# - response.status_code → check for success (200).
# - time.sleep(2) → ensures model loads before proceeding.

def switch_sd_checkpoint(model_name: str, api_url: str = SD_API_URL) -> bool:
    options_endpoint = f"{api_url}/sdapi/v1/options"
    switch_payload = {"sd_model_checkpoint": model_name}
    print(f"🔄 Switching model to: {model_name}")
    try:
        response = requests.post(options_endpoint, json=switch_payload, timeout=30)
        if response.status_code == 200:
            print("✅ Model switch successful.")
            time.sleep(2)
            return True
        else:
            print(f"❌ API Call Failed. Status Code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection/Request Error: {e}")
        return False


# ==== Utility Function: Parse Prompt Output ====
# Purpose: Splits LLM output into positive and negative prompts.
# Why: Stable Diffusion benefits from explicit negative prompts to avoid artifacts.
# Technique: String splitting by common delimiters ("Negative Prompt:").
# Recruiter insight: Shows robustness in handling varied LLM output formats.

def parse_prompt_output(prompt_string: str) -> Tuple[str, str]:
    delimiters = ["NEGATIVE PROMPT:", "Negative Prompt:", "NEGATIVE:", "Negative:"]
    for delimiter in delimiters:
        if delimiter in prompt_string:
            parts = prompt_string.split(delimiter, 1)
            positive = parts[0].strip().strip(',').strip()
            negative = parts[1].strip().strip(',').strip()
            return positive, negative
    print("⚠️ Warning: Could not find negative prompt delimiter in LLM output.")
    return prompt_string, ""


# ==== Step 3: Generate Enhanced Prompt with Ollama ====
# Purpose: Transform a simple idea into a detailed prompt with positive/negative sections.
# Why: Demonstrates integration of LLMs with generative models.
# Syntax:
# - requests.post(OLLAMA_URL, json=ollama_data) → calls Ollama API.
# - ollama_response.json() → parse JSON response.
# Error handling: ConnectionError, Timeout → shows production‑ready resilience.

print("=" * 70)
print("🤖 STEP 1: Generating Enhanced Prompt with Ollama")
print("=" * 70)

base_prompt = (
    "A portrait of a young woman, golden hour sunlight, soft focus, vivid colors, intricate details. "
    "Generate a Stable Diffusion prompt that includes a detailed positive prompt and a section for a "
    f"negative prompt with these keywords: {NEGATIVE_PROMPT_KEYWORDS}. "
    "Ensure the negative section is clearly labeled 'Negative Prompt:'"
)

ollama_data = {
    "model": "brxce/stable-diffusion-prompt-generator",
    "prompt": base_prompt,
    "format": "json",
    "stream": False
}

enhanced_prompt = None
positive_prompt = ""
negative_prompt = NEGATIVE_PROMPT_KEYWORDS

try:
    ollama_response = requests.post(OLLAMA_URL, json=ollama_data, timeout=TIMEOUT_OLLAMA)
    if ollama_response.status_code == 200:
        prompt_json = ollama_response.json()
        if 'response' in prompt_json:
            enhanced_prompt = prompt_json.get("response")
            positive_prompt, negative_prompt = parse_prompt_output(enhanced_prompt)
            print("✅ Ollama Prompt Generation Success!")
        else:
            print("❌ Ollama response missing 'response' key.")
    else:
        print(f"❌ Ollama API call failed (Status: {ollama_response.status_code})")
except requests.exceptions.ConnectionError:
    print("❌ FATAL ERROR: Could not connect to Ollama.")
except requests.exceptions.Timeout:
    print("❌ Timeout: Ollama took too long to respond.")

if not positive_prompt:
    positive_prompt = "A masterwork portrait of a young woman, volumetric golden hour light, highly detailed face"
    negative_prompt = NEGATIVE_PROMPT_KEYWORDS
    print("⚠️ Using FALLBACK Prompts.")


# ==== Step 4: Switch Active Model ====
print("\n" + "=" * 70)
print("🔄 STEP 2: Switching Stable Diffusion Model")
print("=" * 70)

success = switch_sd_checkpoint(MODEL_TO_USE)
if not success:
    print("⚠️ Warning: Model switch failed. Proceeding anyway...")


# ==== Step 5: Generate Image via Stable Diffusion ====
# Purpose: Generate an image using txt2img API with enhanced prompts.
# Key parameters:
# - steps (25): More steps = higher quality, slower speed.
# - cfg_scale (7): Prompt adherence (5-15 typical)
# - sampler_index (Euler a): Sampling algorithm
# - width (768): Must be multiple of 64
# - height (512): Must be multiple of 64
# - batch_size (1): Number of images to generate (1 for single image)
