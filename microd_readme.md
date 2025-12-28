# Micro Distillery App Guide
*UNDER DEVELOPMENT FOR TESTING*

## Overview
Web-based PWA for simulating GRPO + VAE model distillation. Exports models to Hugging Face, SafeTensors, GGUF, ONNX.

## Use Cases
- **Model Distillation Training**: Simulate GRPO optimization with VAE filtering for small LLMs (42M-345M params).
- **Policy Experimentation**: Test group sizes, KL penalties, cache reuse for RLHF-like training.
- **VAE Filtering**: Apply latent space compression to improve distillation quality.
- **Sandbox Testing**: Execute safe Python code with feedback masking.
- **Export & Deployment**: Generate deployable models for inference in various frameworks.
- **Offline Usage**: PWA supports offline training simulation and exports.

## App Usage
1. Open `microd.html` in browser (Chrome recommended for PWA).
2. Configure GRPO: Set group size (4-32), KL penalty (0.01-0.5), etc.
3. Initialize GRPO: Click "INITIALIZE GRPO SYSTEM".
4. Train GRPO: Click "START GRPO TRAINING" (simulates 100 steps).
5. Configure VAE: Set latent dim (8-128), beta (0.001-0.1).
6. Train VAE: Click "TRAIN VAE FILTER" (simulates 50 epochs).
7. Use Sandbox: Enter Python code, click "Execute" (safe, simulated).
8. Monitor: View terminal logs, metrics, tokens.
9. Export: Click "EXPORT TRAINED MODEL", select format/options, export.

Install as PWA for offline: Click install prompt or add to home screen.

## Exporting Models
- After training, open export modal.
- Select format (Hugging Face default).
- Check options: tokenizer, config, quantization (4-bit).
- Click export: Downloads ZIP or files with config.json, pytorch_model.bin, etc.
- Dynamic files update with current configs (e.g., group_size in config.json).

## Using Models on Hugging Face
1. Upload exported folder to HF repo (e.g., `your-username/micro-distill-grpo-vae`).
2. Include generated README.md for details.
3. Load in Python:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model = AutoModelForCausalLM.from_pretrained("your-username/micro-distill-grpo-vae")
   tokenizer = AutoTokenizer.from_pretrained("your-username/micro-distill-grpo-vae")

   inputs = tokenizer("Hello, world!", return_tensors="pt")
   outputs = model.generate(**inputs, max_length=50)
   print(tokenizer.decode(outputs[0]))
   ```
- Custom configs: GRPO/VAE params in config.json for fine-tuning.
- Quantized (4-bit): Use with GPTQ or bitsandbytes.
- Inference: Supports top-k/p sampling; adjust generation_config.json.

## License
Apache 2.0
*UNDER DEVELOPMENT FOR TESTING*
