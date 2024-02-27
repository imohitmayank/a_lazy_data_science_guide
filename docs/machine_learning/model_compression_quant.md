## Introduction

- Quantization is a technique that has been used in digital computing for a long time. It involves compressing data by converting a continuous signal or data set into a discrete set of values or levels. 
- Neural Networks (NNs) present unique challenges and opportunities in the context of quantization. Firstly, both inference and training of NNs require significant computational resources, making the efficient representation of numerical values crucial. Secondly, many current NN models are highly over-parameterized, allowing scope for techniques that could reduce bit precision without sacrificing accuracy. 
- However, an important distinction is that NNs exhibit remarkable resilience to aggressive quantization and extreme discretization. That said, by moving from floating-point representations to low-precision fixed integer values represented in four bits or less, it is possible to significantly reduce memory footprint and latency. In fact, reductions of 4x to 8x are often observed in practice in these applications. This article serves as a beginner-friendly introduction to quantization in deep learning.

## Data Types and Representations

Before diving into the topic, let’s understand the importance and advantages of using smaller data-type representations. 

- Neural Nets consists of weights which are matrices of numbers, where each number is mostly represented in `float32` data type. This means each number’s size is 32 bits (4 bytes) and an average-sized LLM of 7B parameters (like LLaMA) will have a size around 7 * 10^9 * 4  = 28GB! This is vRAM required just for inference, and for training, you might need 2x more memory as the system needs to store gradients as well. *(For finetuning, the memory requirements depend on which optimizer we are using. AdamW needs 8 bytes per parameter)*. Now if we can use half-precision (`float16`) our memory requirements are reduced by half and for much advanced 8-bit representation it becomes just 1/4th of the original requirement! 
- Below is a table with different data types, their ranges, size and more details. 

| Data Type | Min | Max | Range | Bits | Accumulation Data Type |
| --- | --- | --- | --- | --- | --- |
| uint8 | 0 | 255 | 0-255 | 8 | uint16 |
| int8 | -128 | 127 | -128 to 127 | 8 | int16 |
| uint16 | 0 | 65535 | 0-65535 | 16 | uint32 |
| int16 | -32768 | 32767 | -32768 to 32767 | 16 | int32 |
| uint32 | 0 | 4294967295 | 0-4294967295 | 32 | uint64 |
| int32 | -2147483648 | 2147483647 | -2147483648 to 2147483647 | 32 | int64 |
| uint64 | 0 | 18446744073709551615 | 0-18446744073709551615 | 64 | uint64 |
| int64 | -9223372036854775808 | 9223372036854775807 | -9223372036854775808 to 9223372036854775807 | 64 | int64 |
| float16 | -65504 | 65504 | -65504 to 65504 | 16 | float32 |
| float32 | -3.4028235E+38 | 3.4028235E+38 | -3.4028235E+38 to 3.4028235E+38 | 32 | float64 |
| float64 | -1.7976931348623157E+308 | 1.7976931348623157E+308 | -1.7976931348623157E+308 to 1.7976931348623157E+308 | 64 | float128 |

!!! Note
    `float128` isn't a standard data type in many environments, and the accumulation type for some might vary based on the context or platform. For float types, the range values typically represent the maximum magnitude, not the precise range of normal numbers. Please verify against your specific environment or programming language for the most accurate information.

## Basics of Quantizations

Now we are ready to tackle the basic concepts of Quantization in Deep Learning.

### Uniform vs Non-Uniform Quantization

- A normal quantization function is shown below where \( S \) is a scaling factor, \( Z \) is an integer zero point, and \( \text{Int} \) represents an integer mapping through rounding. In the scaling factor \( S \), $[\alpha, \beta]$ denotes the clipping range i.e. a bounded range that we are clipping the real values with, and $b$ is the quantization bit width. The key characteristic of uniform quantization is that the quantized values are evenly spaced. This spacing can be visualized in a graph where the distance between each quantized level is constant.

$$
Q(r) = \text{Int}\left(\frac{r}{S}\right) - Z;
$$

$$
S = \frac{\beta - \alpha}{2^b - 1},
$$

- In contrast to uniform quantization, non-uniform quantization methods produce quantized values that are not evenly spaced. Non-uniform quantization can be more efficient in representing values with a non-linear distribution. However, implementing non-uniform quantization schemes efficiently on general computation hardware (e.g., GPU and CPU) is typically challenging. Therefore, uniform quantization is currently the most commonly used method due to its simplicity and efficient mapping to hardware.

<figure markdown> 
    ![](../imgs/ml_modelcompression_quant1.png){ width="500" }
    <figcaption>Source: [1]</figcaption>
</figure>

- Also note that for both uniform and non-uniform quantization, the original real values can be approximated through dequantization, using the inverse operation \( \tilde{r} = S(Q(r) + Z) \). However, due to the rounding inherent in the quantization process, the recovered values \( \tilde{r} \) will not be exactly the same as the original \( r \). This approximation error is a trade-off for the benefit of reduced precision and computational complexity.

### Symmetric vs. Asymmetric Quantization

<figure markdown> 
    ![](../imgs/ml_modelcompression_quant2.png)
    <figcaption>Source: [1]</figcaption>
</figure>

- In symmetric quantization, the scaling factor \( S \) is determined using a symmetric clipping range, typically defined as \( \alpha = -\beta \). The value for \( \alpha \) and \( \beta \) is often selected based on the maximum absolute value in the data, resulting in \( -\alpha = \beta = \max(|r_{\max}|, |r_{\min}|) \). Symmetric quantization simplifies the quantization process by setting the zero point \( Z \) to zero, thus the quantization equation becomes \( Q(r) = \text{Int}\left(\frac{r}{S}\right) \). There are two versions of symmetric quantization: full range, which utilizes the entire representable range of the data type (e.g., INT8), and restricted range, which excludes the extremes for better accuracy. Symmetric quantization is preferred for weight quantization in practice due to computational efficiency and straightforward implementation.

- Asymmetric quantization uses the actual minimum and maximum values of the data as the clipping range, i.e., \( \alpha = r_{\min} \) and \( \beta = r_{\max} \), resulting in a non-symmetric range where \( -\alpha \neq \beta \). This method may provide a tighter clipping range which is advantageous when the data distribution is imbalanced, such as activations following a ReLU function. Asymmetric quantization allows for a more precise representation of the data's distribution, but at the cost of a more complex quantization process due to the non-zero zero point.

!!! Hint

    Both symmetric and asymmetric quantization require calibration, which involves selecting the appropriate clipping range. A common method is to use the min/max values of the signal; however, this can be susceptible to outliers which may expand the range unnecessarily and reduce quantization resolution. Alternative methods include using percentiles or optimizing for the minimum Kullback-Leibler divergence to minimize information loss.

### Dynamic vs Static Quantization

So far we have discussed about calibrating the clipping range for weights which is relatively simple as it does not change during inference. Calibrating the activations is different as its range could be different for different input. Let's look into different ways to handle it, 

- Dynamic quantization involves calculating the clipping range (\([α, β]\)) in real-time for each activation map based on the current input. Since activation maps change with each new input sample, dynamic range calibration allows the quantization process to adapt to these changes, potentially leading to higher accuracy. The trade-off, however, is the computational overhead required to compute signal statistics on the fly for every input during runtime.

- Static quantization, in contrast, involves determining a fixed clipping range prior to inference. This range is computed using a series of calibration inputs to estimate the typical range of activations. The advantage of this approach is the elimination of computational overhead during inference, as the range is not recalculated for each input. While typically less accurate than dynamic quantization due to its non-adaptive nature, static quantization benefits from methods that optimize the range, such as minimizing the Mean Squared Error between the original and quantized distributions. Other metrics like entropy can also be used, but MSE remains the most popular.

### Quantization Granularity

Quantization granularity refers to the level of detail at which the clipping range \([α, β]\) is determined for quantization. There are various levels at which this can be implemented:

- **Layerwise quantization** sets a single clipping range based on the collective statistics of all the weights in a layer's convolutional filters. This method is straightforward to implement but may lead to suboptimal accuracy. The reason is that different filters within the layer can have widely varying ranges, and using a single range for all may compromise the resolution of filters with narrower weight ranges.

- **Groupwise quantization** segments multiple channels within a layer and calculates a clipping range for each group. This method can be beneficial when parameter distributions vary significantly within the layer, allowing for more tailored quantization. However, managing multiple scaling factors adds complexity.

- **Channelwise quantization** assigns a unique clipping range and scaling factor to each channel or convolutional filter. This granularity level is widely adopted because it provides a high quantization resolution and often yields higher accuracy without significant overhead.

- **Sub-channelwise quantization** further divides the granularity to smaller groups within a convolution or fully-connected layer. Although it could potentially lead to even higher accuracy due to finer resolution, the computational overhead of managing numerous scaling factors is considerable. Therefore, while channelwise quantization is a standard practice, sub-channelwise is not, due to its complexity and overhead.

<figure markdown> 
    ![](../imgs/ml_modelcompression_quant3.png)
    <figcaption>Source: [1]</figcaption>
</figure>

## Different Types of Quantization

Below are three primary types of quantization methods used in neural networks:

1. **Quantization-Aware Training (QAT):**

   - Quantization may skew the weights by moving them away from their converged points. To mitigate this, in QAT the model is retrained with quantized parameters to converge to a new optimal point. This involves using a forward and backward pass on a quantized model but updating the model parameters in floating-point precision. After each gradient update, the model parameters are quantized again. QAT utilizes techniques such as the Straight Through Estimator (STE) to approximate the gradient of the non-differentiable quantization operator. Other approaches like regularization operators or different gradient approximations are also explored.

     - **Advantages:** QAT typically results in models with better performance and minimal accuracy loss due to the careful retraining with quantized parameters.

     - **Disadvantages:** It is computationally expensive as it involves retraining the model, often for several hundred epochs.

2. **Post-Training Quantization (PTQ):**
   - PTQ is applied after a model has been trained with full precision. It adjusts the weights and activations of a model without any retraining or fine-tuning. Various methods exist to mitigate accuracy loss in PTQ, including bias correction methods, optimal clipping range calculations, outlier channel splitting, and adaptive rounding methods.
     - **Advantages:** PTQ is a quick and often negligible overhead method for reducing the size of neural network models. It is particularly useful when training data is limited or unavailable.
     - **Disadvantages:** Generally, PTQ leads to lower accuracy compared to QAT, particularly for low-precision quantization.

    !!! Note
        While no model finetuning happens in PTQ, we may use training dataset for activation calibration.

1. **Zero-shot Quantization (ZSQ):**
   - ZSQ refers to performing quantization without any access to the training or validation data. This is particularly vital for scenarios where the dataset is too large, proprietary, or sensitive. Approaches to ZSQ include generating synthetic data that closely mimics the real data distribution using techniques like Generative Adversarial Networks (GANs) or utilizing the statistics stored in batch normalization layers.
     - **Advantages:** ZSQ is crucial for scenarios where data privacy or availability is a concern. It allows the quantization of models without needing access to the original dataset.
     - **Disadvantages:** While innovative, ZSQ methods may not capture the nuances of the actual data distribution as effectively as methods with access to real data, potentially leading to less accurate models.

In summary, each quantization method has its own set of trade-offs between accuracy, efficiency, and applicability. The choice among QAT, PTQ, and ZSQ depends largely on the specific constraints of the deployment environment, the availability of computational resources, and the necessity for data privacy.

## Quantization in Practice

In practice, PTQ *(Post-Training Quantization)* is one of the most widely used quantization methods due to its simplicity and minimal overhead. It is particularly effective for reducing the size of neural network models without requiring access to the original training data. Here is the process of using PTQ based techniques for quantizing a model:

1. First we get the model which is trained in full precision *(float32)*.
2. Next, we can either quantized the model and save it or we can quantize the model on the fly during inference. 

!!! Note
    We need to dequant the model *(convert back to higher precision)* during inference. This is because inference requires forward pass which consists of complex computations like matrix multiplication and currently float-float matmul is much faster than int-int matmul. *([Refer](https://stackoverflow.com/questions/45373679/why-is-it-faster-to-perform-float-by-float-matrix-multiplication-compared-to-int))*

!!! Hint
    You can find thousands of quantized models *(different formats)* on the [TheBloke's collection](https://huggingface.co/TheBloke) in HuggingFace.


<!-- ### AQLM 

- Feb 2024, latest -->

### AWQ

- Activation-aware Weight Quantization (AWQ), introduced in Oct 2023, is a Weight only quantization method based on the fact that not all weights are equally important for the model's performance. With this in mind, AWQ tries to identify those salient weights using the activation distribution where weights with larger activation magnitudes are deemed crucial for model performance. On further analysis, it was found that just a minor fraction (~1%) of weights, if left unquantized (FP16) could lead to non-significant change in model performance. While this is a crucial observation, it is also important to note that partial quantization of weights leads to mixed-precision data types, which are not efficiently handled in many hardware architectures. To circumvent these complexities, AWQ introduces a novel per-channel scaling technique that scales the weights *(multiple weight by scale $s$ and inverse scale the activation i.e multiple activation by $1/s$)* before quantization, where $s$ is usually greater than 1, and it is determined by a grid search. This minor trick optimizes the quantization process, removes the need for mixed-precision data types, and keep the performance consistent with 1% FP16 weights.
  
<figure markdown> 
    ![](../imgs/ml_modelcompression_quant_awq.png)
    <figcaption>Source: [3]</figcaption>
</figure>

- Empirical evidence demonstrates AWQ's superiority over existing quantization techniques, achieving remarkable speedups and facilitating the deployment of large models on constrained hardware environments. Notably, AWQ has enabled the efficient deployment of massive LLMs, such as the Llama-2-13B model, on single GPU platforms with limited memory (~8GB), and has achieved significant performance gains across diverse LLMs with varying parameter sizes. 

<figure markdown> 
    ![](../imgs/ml_modelcompression_quant_awq2.png)
    <figcaption>Better Perplexity score of AWQ on LLaMA-1 and 2 models in comparison with other quantization techniques. Source: [3]</figcaption>
</figure>

- Now, running inference on AWQ model is made simlper by using the `transformers` library. Below is an example of how to use AWQ model for inference.

```python
# install
# !pip install autoawq

# import
from transformers import AutoModelForCausalLM, AutoTokenizer
# one sample AWQ quantized model
model_id = "TheBloke/zephyr-7B-alpha-AWQ"
# load model
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0")
```

<!-- ### GPTQ

thereby maintaining the model's generalization capabilities across various domains without the risk of overfitting to specific calibration sets. -->

## References

[1] [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630)

[2] Maarten Grootendorst's Blog - [Which Quantization Method is Right for You? (GPTQ vs. GGUF vs. AWQ)](https://www.maartengrootendorst.com/blog/quantization/)

[3] AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration - [Paper](https://arxiv.org/abs/2306.00978) | [Code](https://github.com/mit-han-lab/llm-awq)