# VIdeo-Summariser
Web application powered by OpenAI's Whisper and pretrained BART

## Getting Started

### Dependencies and Installation

```
# clone this repository

# create new conda environment
conda create --name video_summariser python=3.10
conda activate video_summariser

# install dependcies
pip install -r requirements.txt

# Pytorch is optional for GPU accelarted version of the application
# follow the directions in: https://pytorch.org/get-started/locally/

```

### Executing program

* Execute the program by running:
```
streamlit run web.py/web_gpu.py
```


