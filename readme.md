# LLM Model Converter and Quantizer

Large Language Models (LLMs) are typically distributed in formats optimized for training (like PyTorch) and can be extremely large (hundreds of gigabytes), making them impractical for most real-world applications. This tool addresses two critical challenges in LLM deployment:

1. **Size**: Original models are too large to run on consumer hardware
2. **Format**: Training formats are not optimized for inference
   
![Quantization](Quantization.gif)


## Why This Tool?

I decided to build this tool to help AI Researchers achieve the following:
- Converting models from Hugging Face to GGUF format (optimized for inference)
- Quantizing models to reduce their size while maintaining acceptable performance
- Making deployment possible on consumer hardware (laptops, desktops) with limited resources

### The Problem
- LLMs in their original format require significant computational resources
- Running these models typically needs:
  - High-end GPUs
  - Large amounts of RAM (32GB+)
  - Substantial storage space
  - Complex software dependencies

### The Solution
This tool provides:
1. **Format Conversion**
   - Converts from PyTorch/Hugging Face format to GGUF
   - GGUF is specifically designed for efficient inference
   - Enables memory mapping for faster loading
   - Reduces dependency requirements

2. **Quantization**
   - Reduces model size by up to 4-8x
   - Converts from FP16/FP32 to more efficient formats (INT8/INT4)
   - Maintains reasonable model performance
   - Makes models runnable on consumer-grade hardware

3. **Accessibility**
   - Enables running LLMs on standard laptops
   - Reduces RAM requirements
   - Speeds up model loading and inference
   - Simplifies deployment process

## 🎯 Purpose

This tool helps developers and researchers to:
- Download LLMs from Hugging Face Hub
- Convert models to GGUF (GPT-Generated Unified Format)
- Quantize models for efficient deployment
- Upload processed models back to Hugging Face

## 🚀 Features

- **Model Download**: Direct integration with Hugging Face Hub
- **GGUF Conversion**: Convert PyTorch models to GGUF format
- **Quantization Options**: Support for various quantization levels
- **Batch Processing**: Automate the entire conversion pipeline
- **HF Upload**: Option to upload processed models back to Hugging Face

# Quantization Types Overview

| **Quantizer Name** | **Purpose**                                            | **Benefits**                                                    | **When to Use**                                            |
|--------------------|--------------------------------------------------------|-----------------------------------------------------------------|------------------------------------------------------------|
| **Q2_K**           | Quantizes model to 2 bits using K mode                 | Minimizes memory usage, faster inference                       | Use for highly memory-constrained environments            |
| **Q3_K_l**         | 3-bit quantization using low precision mode            | Balance between size reduction and inference quality            | When a small model size with moderate precision is needed  |
| **Q3_K_M**         | 3-bit quantization with medium precision mode          | Better performance with slight increase in memory usage         | When moderate precision and size reduction are desired    |
| **Q3_K_S**         | 3-bit quantization using high precision mode           | Higher inference quality with minimal size reduction            | When inference quality is a higher priority than size     |
| **Q4_0**           | 4-bit quantization with zero mode                      | Reduced model size with minimal impact on performance           | Use when a larger model is required but memory is limited |
| **Q4_1**           | 4-bit quantization with another precision mode         | Better performance than Q4_0 with slight increase in size       | When a balance of size and performance is required        |
| **Q4_K_M**         | 4-bit quantization using K mode with medium precision  | Further optimized performance with reduced model size           | For performance optimization in moderately sized models   |
| **Q4_K_S**         | 4-bit quantization using K mode with high precision    | Optimized for size with higher precision                        | When slightly higher precision and smaller size are needed|
| **Q5_0**           | 5-bit quantization using zero mode                     | Larger model size with enhanced precision                       | Use when memory is not a major constraint and high precision is required |
| **Q5_1**           | 5-bit quantization with an alternative mode            | Offers trade-off between size and performance                   | For improved performance at the cost of some additional memory usage |
| **Q5_K_M**         | 5-bit quantization using K mode with medium precision  | Better model compression and performance                        | When model performance is crucial and space is a concern  |
| **Q5_K_S**         | 5-bit quantization using K mode with high precision    | Optimal performance with minimal size increase                  | Use for high-performance applications with moderate memory limits |
| **Q6_K**           | 6-bit quantization using K mode                         | Larger model size but better precision                          | For applications where precision is critical and space is more available |
| **Q8_0**           | 8-bit quantization with zero mode                      | Maximum size reduction with reasonable precision                | Use when model size is most critical and higher precision is not needed |
| **BF16**           | 16-bit Brain Floating Point quantization               | Balances precision and size with higher performance             | When a high level of performance is needed with moderate memory usage |
| **F16**            | 16-bit Floating Point quantization                     | Offers good precision and performance with moderate memory usage | When maintaining a high precision model is essential     |
| **F32**            | 32-bit Floating Point quantization                     | Highest precision, best for model training and inference        | Use when maximum precision is required for sensitive tasks |


## 💡 Why GGUF?

GGUF (GPT-Generated Unified Format) offers several advantages:

# GGUF (GPT-Generated Unified Format)

GGUF (GPT-Generated Unified Format) is a file format specifically designed for efficient deployment and inference of large language models. Let me break down why it's important and beneficial:

## Key Benefits of GGUF:

### Optimized for Inference:
- GGUF is specifically designed for model inference (running predictions) rather than training.
- It's the native format used by llama.cpp, a popular framework for running LLMs on consumer hardware.

### Memory Efficiency:
- Reduces memory usage compared to the original PyTorch/Hugging Face formats.
- Allows running larger models on devices with limited RAM.
- Supports various quantization levels (reducing model precision from FP16/FP32 to INT8/INT4).

### Faster Loading:
- Models in GGUF format can be memory-mapped (mmap), meaning they can be loaded partially as needed.
- Reduces initial loading time and memory overhead.

### Cross-Platform Compatibility:
- Works well across different operating systems and hardware.
- Doesn't require Python or PyTorch installation.
- Can run on CPU-only systems effectively.

### Embedded Metadata:
- Contains model configuration, tokenizer, and other necessary information in a single file.
- Makes deployment simpler as all required information is bundled together.


## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/bhaskatripathi/LLM_Quantization

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

```bash
# Run the Streamlit application
streamlit run app.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see below for details:

## ⚠️ Requirements

- Python 3.8+
- Streamlit
- Hugging Face Hub account (for model download/upload)
- Sufficient storage space for model processing

## 📚 Supported Models

The tool currently supports various model architectures including:
- DeepSeek models
- Mistral models
- Llama models
- Qwen models
- And more...

## 🤔 Need Help?

If you encounter any issues or have questions:
1. Check the existing issues
2. Create a new issue with a detailed description
3. Include relevant error messages and environment details

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the model hub
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF format implementation
- All contributors and maintainers

---
Made with ❤️ for the AI community
