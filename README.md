The primary goal of this project is to refine the `stable-code-3b` model, a transformer language model with 2.7 billion parameters, specifically engineered for code comprehension and generation across diverse programming languages.

A significant challenge within this project is adapting the model to process new data efficiently while maintaining optimal performance and accuracy. The utilized dataset consists of large-scale, open-source textual and code data, which has been meticulously filtered and tailored for this task. This initiative aims to enhance code generation capabilities by methodically adjusting the model's weights using specific fine-tuning scripts. 

A particularly noteworthy challenge was to confine the fine-tuning process to the capabilities of a GeForce RTX 3080 graphics card, equipped with 10GB of VRAM, thus enabling a completely local setup without reliance on cloud-based infrastructure. Adapting such a substantial model within limited GPU memory necessitated innovative strategies and optimizations.

A series of strategic decisions were undertaken to address the model's size and ensure successful fine-tuning within constrained computational environments:

- **LoRA Technique**: The Low-Rank Adaptation (LoRA) method focused updates on select model components, reducing parameter numbers and computational demands. This facilitated efficient memory use on available hardware during fine-tuning.

- **Quantization and 4bit Precision**: The deployment of 8bit precision markedly minimized VRAM consumption, enabling the large model to function on a GeForce RTX 3080, which possesses restricted VRAM capacity.

- **Freezing Layers Technique**: By immobilizing all but the final model layers, we significantly decreased trainable parameter counts, aligning resources with the PEFT approach.

- **PYTORCH_CUDA_ALLOC_CONF**: The configuration of this environment variable (`max_split_size_mb:128`) strategically handled memory fragmentation in CUDA allocations, mitigating potential memory limitation issues.

- **Accelerate Library**: Utilized within the current framework, it presents a huge improvement for reducing memory overhead by offloading computations and memory usage to CPU + RAM if needed
  
Faced with the primary constraint of localizing the fine-tuning process of a large model within the confines of a GeForce RTX 3080 (10GB VRAM), these techniques collectively facilitated substantial advancements. By strategically managing computational and memory demands, the project achieved comprehensive local model training, thereby broadening access to state-of-the-art AI technologies without cloud dependencies. Such advancements demonstrate the feasibility of employing advanced models on limited resource setups, opening avenues for widespread application in diverse settings.
