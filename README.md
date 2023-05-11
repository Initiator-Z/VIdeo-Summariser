# Video-Summariser
Web application powered by OpenAI's Whisper and pretrained BART

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


