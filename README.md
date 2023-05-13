# Video-Summariser
Web application powered by OpenAI's Whisper and pretrained BART

## Description
This web application performs videos transcribing and summarisation for a give video file. 
The application uses Whisper, an automatic speech recognition model to transcribe the video's audio content into text. The text is then summarised using BART, a pretrained transformer to generate a concise and accurate summary of the video's content. The original video and the summary are displayed side by side allowing users to easily access both versions of the content. Options are available for chosing various size of Whisper model and finetued BART model. 

## Getting Started

### Dependencies and Installation
* Pytorch and cuda is optional for GPU accelerated version of the application\
Follow the directions in: https://pytorch.org/get-started/locally/

```
# clone this repository
git clone https://github.com/Initiator-Z/Video-Summariser.git
cd Video-Summariser

# create new conda environment
conda create --name video_summariser python=3.10
conda activate video_summariser

# install dependcies
pip install -r requirements.txt
```


### Finetuned Model
* Download optional model finetuned with TED Talks and unzip to the same directory as the py files
https://drive.google.com/file/d/1X7HAtapky6u9HZq1-nDQu7tQ0kXJ3F3F/view?usp=share_link

### Executing program

* Execute the program by running:
```
streamlit run web.py/web_gpu.py
```

## Citations
1. OpenAI. (2023). Whisper. GitHub. https://github.com/openai/whisper
2. Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. CoRR, abs/1910.13461. http://arxiv.org/abs/1910.13461

