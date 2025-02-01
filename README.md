<h1 align="center">Distillation and Pruning for Scalable Self-Supervised Representation-Based Speech Quality Assessment</h1>

This repository contains sample code and weights accompanying the paper "Distillation and Pruning for Scalable Self-Supervised Representation-Based Speech Quality Assessment". Distill-MOS is an compact and efficient speech quality assessment model learned from a larger SSL-based speech quality assessment model.

## Usage
### Local Installation

To use the model locally, simply install using pip:

```bash
pip install distillmos
```

### Sample Inference Code
Model instantiation is as easy as:
```python
import distillmos

sqa_model = distillmos.ConvTransformerSQAModel()
sqa_model.eval()
```
Weights are loaded automatically.

The input to the model is a `torch.Tensor` with shape `[batch_size x signal_length]`, containing mono speech waveforms **sampled at 16kHz**. The model returns a batch of estimated mean opinion scores in the range 1 (bad) .. 5 (excellent).
```python
import torchaudio

x, sr = torchaudio.load('my_speech_file.wav')
if x.shape[0] > 1:
    print(
        f"Warning: file has multiple channels, using only the first channel."
    )
x = x[0, None, :]

# resample to 16kHz if needed
if sr != 16000:
    x = torchaudio.transforms.Resample(sr, 16000)(x)

with torch.no_grad():
    mos = sqa_model(x)

print('MOS Score:', mos)
```

### Command Line Interface
You can also use distillmos from the command line for inference on single .wav files, folders containing .wav files, and lists of file paths. Please call
```bash
distillmos --help
```
for a detailed list of available commands and options.

## Example Ratings
Here are some example estimates from the <a href="https://github.com/QxLabIreland/datasets/tree/597fbf9b60efe555c1f7180e48a508394d817f73/genspeech">GenSpeech</a> and <a href="https://github.com/QxLabIreland/datasets/tree/597fbf9b60efe555c1f7180e48a508394d817f73/tcdvoip/TCDvoip/Test_Set/NOISE">TCDvoip-NOISE</a> datasets.


### GenSpeech
Example: LPCNet_listening_test/mfall/dir1/

**LPCNet Quantized (Distill-MOS: ??, Human MOS: ??)**  
https://github.com/user-attachments/assets/2ef36e8c-6915-4e4e-9697-871a6420da5a

**LPCNet Unquantized (Distill-MOS: ??, Human MOS: ??)**  
https://github.com/user-attachments/assets/1b08cd9d-c395-46c2-8707-c216bd742320

**MELP (Distill-MOS: ??, Human MOS: ??)**  
https://github.com/user-attachments/assets/6fee729d-469e-4dc8-a5cd-667637411974

**Opus (Distill-MOS: ??, Human MOS: ??)**  
https://github.com/user-attachments/assets/fecdf175-0e32-458a-8e20-1861537d110a

**Speex (Distill-MOS: ??, Human MOS: ??)**  
https://github.com/user-attachments/assets/df61b5d9-5c69-472a-a6c7-05d331785b55

**Reference (Distill-MOS: ??, Human MOS: ??)**  
https://github.com/user-attachments/assets/1707455f-ead8-41a7-81c8-98b062229c3d


### TCDvoip-NOISE  
Speaker MK 

**C01 (Distill-MOS: ??, Human MOS: ??)**  
<video controls>
  <source src="./test_audio/TCDvoip-NOISE/C_01_NOISE_MK.wav" type="audio/wav">
</video>

**C05 (Distill-MOS: ??, Human MOS: ??)**  
<video controls>
  <source src="./test_audio/TCDvoip-NOISE/C_05_NOISE_MK.wav" type="audio/wav">
</video>

**C09 (Distill-MOS: ??, Human MOS: ??)**  
<video controls>
  <source src="./test_audio/TCDvoip-NOISE/C_09_NOISE_MK.wav" type="audio/wav">
</video>

**C13 (Distill-MOS: ??, Human MOS: ??)**  
<video controls>
  <source src="./test_audio/TCDvoip-NOISE/C_13_NOISE_MK.wav" type="audio/wav">
</video>

**C17 (Distill-MOS: ??, Human MOS: ??)**  
<video controls>
  <source src="./test_audio/TCDvoip-NOISE/C_17_NOISE_MK.wav" type="audio/wav">
</video>

**C21 (Distill-MOS: ??, Human MOS: ??)**  
<video controls>
  <source src="./test_audio/TCDvoip-NOISE/C_21_NOISE_MK.wav" type="audio/wav">
</video>


## Intended Uses

### Primary Use Cases

The model released in this repository takes a short speech recording as input and predicts its perceptual speech quality by providing an estimated mean opinion score (MOS). 
The model is much smaller than other state-of-the-art MOS estimators, providing a trade-off between parameter count and speech quality estimation performance.
The primary use of this model is to reproduce results reported in the paper and for research purposes as a relatively light-weight MOS estimation model that generalizes across a variety of tasks. 

### Use Case Considerations and Model Limitations

The model is only evaluated on the tasks reported in the paper, including deep noise suppression and signal improvement, and only for speech. The model may not generalize to unseen tasks or languages. 
Use of the model in unsupported scenarios may result in wrong or misleading speech quality estimates. 
When using the model for a specific task, developers should consider accuracy, safety, and fairness, particularly in high-risk scenarios. 
Developers should be aware of and adhere to applicable laws or regulations (including privacy, trade compliance laws, etc.) that are relevant to their use case.

***Nothing contained in this Model Card should be interpreted as or deemed a restriction or modification to the license the model is released under.*** 


## Benchmarks
<img src="benchmark_results.png" alt="Pearson correlation coefficient on test datasets for baselines, teacher model, and selected distilled and pruned models." width="1000">

## Training


|                     |     |
|---------------------|-----| 
| Developer           | Microsoft |
| Architecture        | Convolutional transformer |
| Inputs              | Speech recording |
| Input length        |  |
| GPUs                | 4 x A6000 |
| Training time       | |
| Training data       | |
| Outputs             | Estimate of speech quality mean-opinion score (MOS) |
| Dates               | Trained between May and July 2024 |
| Status              | |
| Supported languages | English |
| Release date        | Oct 2024 |
| License             | MIT |

### Training Datasets

The model is trained on a large set of speech samples:

- About 2600 hours of unlabeled speech and 180 hours of noise recordings from the <a href="https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-icassp-2022/">ICASSP 2022 Deep Noise Suppression Challenge</a>
- The output of publicly available <a href="https://github.com/coqui-ai/TTS">text-to-speech synthesis models</a>
- <a href="https://www.isca-archive.org/interspeech_2020/mittag20b_interspeech.pdf">PSTN</a>
- <a href="https://github.com/ConferencingSpeech/ConferencingSpeech2022">ConferencingSpeech 2022 Challenge</a>
- <a href="https://www.isca-archive.org/interspeech_2021/mittag21_interspeech.pdf">NISQA</a>
- <a href="https://voicemos-challenge-2022.github.io">VoiceMOS Challenge 2022</a>
- Submissions to <a href="https://www.microsoft.com/en-us/research/uploads/prod/2021/06/0006608.pdf">ICASSP 2021 Deep Noise Suppression Challenge</a>
- Submissions to <a href="https://www.isca-archive.org/interspeech_2022/diener22_interspeech.pdf">Interspeech 2022 audio deep packet loss concealment challenge</a>
- Submissions to <a href="https://www.microsoft.com/en-us/research/academic-program/speech-signal-improvement-challenge-icassp-2023">ICASSP 2023 Speech Signal Improvement Challenge</a>
  
## Responsible AI Considerations

Similarly to other (audio) AI models, the model may behave in ways that are unfair, unreliable, or inappropriate. Some of the limiting behaviors to be aware of include:

* **Quality of Service** and **Limited Scope**: The model is trained primarily on spoken English and for speech enhancement or degradation scenarios. Evaluation on other languages, dialects, speaking styles, or speech scenarios may lead to inaccurate speech quality estimates.

* **Representation and Stereotypes**: This model may over- or under-represent certain groups, or reinforce stereotypes present in speech data. These limitations may persist despite safety measures due to varying representation in the training data.

* **Information Reliability**: The model can produce speech quality estimates that might seem plausible but are inaccurate.

Developers should apply responsible AI best practices and ensure compliance with relevant laws and regulations. Important areas for consideration include:

* **Fairness and Bias**: Assess and mitigate potential biases in evaluation data, especially for diverse speakers, accents, or acoustic conditions.

* **High-Risk Scenarios**: Evaluate suitability for use in scenarios where inaccurate speech quality estimates could lead to harm, such as in security or safety-critical applications.

* **Misinformation**: Be aware that incorrect speech quality estimates could potentially create or amplify misinformation. Implement robust verification mechanisms.

* **Privacy Concerns**: Ensure that the processing of any speech recordings respects privacy rights and data protection regulations.

* **Accessibility**: Consider the model's performance for users with visual or auditory impairments and implement appropriate accommodations.

* **Copyright Issues**: Be cautious of potential copyright infringement when using copyrighted audio content.

* **Deepfake Potential**: Implement safeguards against the model's potential misuse for creating misleading or manipulated content.

Developers should inform end-users about the AI nature of the system and implement feedback mechanisms to continuously improve alignment accuracy and appropriateness.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
