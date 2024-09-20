
- Here are some questions and their answers to make you ready for your next interview. Best of luck 👋

!!! Question ""
    === "Question"
        #### What is the difference between Sample Rate, Bit Depth and Bit Rate?

    === "Answer"

        - **Sample rate** is the number of audio samples recorded per unit of time. For example, an audio with 16kHz sample rate, means that for one second, we have captured 16000 samples. 
        - **Bit Depth** measures how precisely the samples were encoded. Here for a 16kHz sample rate audio, if the bit depth is 8 bits, it means we are using 8 bits of data to store each 16k samples per second.  
        - **Bit rate** is the amount of bits that are recorded per unit of time. For the above example, it means we have `16k * 8 bits` of data per second i.e. `128kbps`

!!! Question ""
    === "Question"
        #### What is the difference between Frame, Frame rate, Number of Channels and Sample size?

    === "Answer"

        - **Frame**: one sample of the audio data per channel.
        - **Frame rate:** the number of times per unit time the sound data is sampled. Same as sample rate.
        - **Number of channels:** indicates if the audio is mono, stereo, or quadro.
        - **The sample size:** the size of each sample in bytes.

!!! Question ""
    === "Question"
        #### What is the difference between i-vector, d-vector and x-vector?

    === "Answer"

        - All of these are vector representation of the audio to capture the speaker information. Let's go through them, 
          - **i-vector** extraction is essentially a dimensionality reduction of the GMM supervector. Refer [SO Question](https://stackoverflow.com/questions/37508698/difference-between-i-vector-and-d-vector)
          - **d-vector** use the Long Short-Term Memory (LSTM) model to the process each individual frame *(along with its context)* to obtain a frame-level embedding, and average all the frame-level embeddings to obtain the segment-level embedding which can be used as the speaker embedding. Refer [paper](https://ieeexplore.ieee.org/document/9054273)
          - **x-vector** take a sliding window of frames as input, and it uses Time Delay Neural Networks (TDNN) to handle the context, to get the frame-level representation. It then has a statistics pooling layer to get the mean and sd of the frame-level embeddings. And then pass the mean and sd to a linear layer to get the segment-level embedding. Refer the [original Paper](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf), [OxfordWaveResearch Slides](https://oxfordwaveresearch.com/wp-content/uploads/2020/02/IAFPA19_xvectors_Kelly_et_al_presentation.pdf) and [post on r/speechtech](https://www.reddit.com/r/speechtech/comments/ogq13y/whats_the_main_difference_between_dvector_and/)
