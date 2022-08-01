!!! warning
    This page is still under progress. Please check back after some time or [contact me](mailto:mohitmayank1@gmail.com)

## Introduction

- [Wav2Vec](https://arxiv.org/abs/2006.11477) is a framework for [self-supervised](../../machine_learning/introduction/#self-supervised-learning) learning of representations from raw audio data. Basically it learns to efficiently represent the raw audio data as a vector using raw audio data.

<figure markdown> 
    ![](../imgs/audio_wav2vec2_arch.png){ width="500" }
    <figcaption>Illustration of the Wav2vec2 framework ([Wav2vec2 paper](https://arxiv.org/abs/2006.11477))</figcaption>
</figure>

- A major advantage of this approach is that we end up training a generic audio model that could be used for multiple downtream tasks! And because of the self-supervised learning, we don't access to huge amount of labeled data. In the paper, after pre-training on unlabeled speech, the model is fine-tuned on labeled data with a [Connectionist Temporal Classification (CTC)](../audio_intelligence/connectionist_temporal_classification.md) loss for [speech recognition task](../audio_intelligence/stt.md).

## Architecture

- The complete architecture of the model can be divided into 4 components, they are
  - **Feature encoder**: This is the encoder part of the model. It takes the raw audio data as input and outputs feature vectors. Input size is limited to 400 samples which is 20ms for 16kHz [sample rate](../audio_intelligence_terms#sample-rate-bit-depth-and-bit-rate). The raw audio is first standardized to have zero mean and unit variance. Then it is passed to 1D convolutional neural network (temporal convolution) followed by layer normalization and GELU activation function. There could be 7 such convolution blocks with constant channel size (512), decreasing kernel width (10, 3x4, 2x2) and stride (5, 2x6). The output is list of feature vectors each with 512 dimensions.
  - **Transformers**: The output of the feature encoder is passed on to a transformer layer. One differentiator is use of relative positional embedding by using convolution layers, rather than using fixed positional encoding as done in original Transformers paper. The block size differs, as 12 transformers block with model dimension of 768 for BASE model but 24 blocks with 1024 dimension for LARGE version. 
  - **Quantization module**: For self-supervised learning, we need to work with discrete outputs. For this, there is a quantization module that converts the continous vector output to discrete representations, and on top of it, it automatically learns the discete speech units. This is done by maintaining multiple codebooks/groups (320 in size) and the units are sampled from each codebook are later concatenated *(320x320=102400 possiblt speech units)*. The sampling is done using Gumbel-Softmax which is like argmax but differentiable. 

## Training

- To pre-train the model, Wav2Vec2 masks certain portions of time steps in the featuree encoder which is similar to masked language model. The aim is to teach the model to predict the correct qunatized latent audio representation in a set of distractors for each time step.
- The overall training objective is to minimize contrastive loss ($L_m$) and diversity loss ($L_d$) in $L = L_m + \alpha L_d$. Contrastive loss is the performance on the self-supervised task. Diversity loss is designed to increase the use of the complete quantized codebook representations, and not only a select subset.
- For pretraining, the datasets used were (1) Librispeech corpus with 960 hours of audio data, (2) LibriVox 60k hours of audio data that was later subset to 53.2k hours. Only unlabeled data was used for pretraining.
- To make the model more robust to different tasks, we can finetune the model on a different task. Here, the paper finetuned for ASR by adding a randomly initialized classification layer on top on Transformer layer with class size equal to the size of vocab. The model is optimized by minimizing the CTC loss. 
- Adam was used as optimization algorithm and the learning rate is warmed up till 10% of the training duration, then help constant for next 40% and finally linearly decayed for the remaining duration. Also, for the first 60k updates only output classifier was trained after which Transformer is also updated. The featue encoder is not trained.


## Results

- There are two interesting points to note from the results of the Wav2Vec,
  - The model is able to learn ASR with as minimum as 10 mins of labeled data! As shown below, $LARGE$ model pre-trained on LV-60k and finetuned on Librispeech with CTC loss is giving 4.6/7/9 WER! This is a very good news incase you want to finetune the model to your domain or accent!
  - The choice of decoder can lead to improvment in performance. As obvious from the results, Transformer decoder is giving best performance, followed by n-gram and then CTC decoding. But also note the CTC decoding will gives the best speech. Finally, the suggested decoder is 4-gram as it provides huge improvement in performance by fixing the spellling mistakes and grammer issues and is still faster than transformer models.


<figure markdown> 
    ![](../imgs/audio_wav2vec2_results1.png){ width="500" }
    <figcaption>WER on Librispeech dev/test data ([Wav2vec2 paper](https://arxiv.org/abs/2006.11477))</figcaption>
</figure>

## Code

### Offline transcription using Wav2Vec2 (CTC)

- Here is the code to perform offline transcription using Wav2Vec2 model with `transformer` package. Note the default decoder is CTC.

``` python linenums="1"
# import 
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# load the tokenizer and model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# load the audio data (use your own wav file here!)
input_audio, sr = librosa.load('my_wav_file.wav', sr=16000)

# tokenize
input_values = tokenizer(input_audio, return_tensors="pt", padding="longest").input_values

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)

# print the output
print(transcription)
```

### Offline transcription using Wav2Vec2 (N-gram)

- There is a pre-trained model in Huggingface with N-gram decoder and n-gram lnaguage model. The usage is very similar to the CTC model, we just have to change the model name. Note, this downloads the Wav2Vec2 model plus the N-gram language model.

``` python linenums="1"
# import
from transformers import Wav2Vec2ProcessorWithLM

# load the processor
processor = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

# load the audio data (use your own wav file here!)
input_audio, sr = librosa.load('my_wav_file.wav', sr=16000)

# tokenize
input_values = tokenizer(input_audio, return_tensors="pt", padding="longest").input_values

# retrieve logits
logits = model(input_values).logits

# decode using n-gram
transcription = processor.batch_decode(logits.numpy()).text

# print the output
print(transcription)
```

<!-- ### Creating your own N-gram language model 

- To use n-gram model we can [KenML](https://github.com/kpu/kenlm) to create language model and then use [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode) for decoding.  -->

### Online transcription using Wav2Vec2

- For live transcription using Wav2Vec2, we can utilize [wav2vec2-live](https://github.com/oliverguhr/wav2vec2-live) package. 
- Once you have cloned the repo and installed the packages from `requirements.txt`, the live transcription can be started with *(taken from the package readme and modified)*, 

``` python linenums="1"
# import
from live_asr import LiveWav2Vec2

# load model
english_model = "facebook/wav2vec2-large-960h-lv60-self"
asr = LiveWav2Vec2(english_model,device_name="default")

# start the live ASR
asr.start()

try:        
    while True:
        text,sample_length,inference_time = asr.get_last_text()                        
        print(f"Duration: {sample_length:.3f}s\tSpeed: {inference_time:.3f}s\t{text}")
        
except KeyboardInterrupt:   
    asr.stop()  
```

- This starts the Live ASR on your terminal. The code listen to the audio in your microphone, identifies the chunks with voice using VAD and then pass the voiced chunks to Wave2Vec2 for transcription. One sample output is shown below, 

``` shell 
listening to your voice

Duration: 0.780s	Speed: 0.205s	hello
Duration: 0.780s	Speed: 0.190s	hello
Duration: 0.960s	Speed: 0.223s	my name
....
```


## Additional Materials

- [An Illustrated Tour of Wav2vec 2.0 by Jonathan Bgn](https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html)
- [Boosting Wav2Vec2 with n-grams in 🤗 Transformers](https://huggingface.co/blog/wav2vec2-with-ngram)