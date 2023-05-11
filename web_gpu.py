import os
import torch
import whisper
import streamlit as st
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer

# Let user select the video to be processed and save to current directory if not already
def save_file(file_data, filename):
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        st.warning("File already exists in the current directory")
    else:
        with open(file_path, "wb") as f:
            f.write(file_data.getbuffer())
        st.success("File saved to the current directory for further processing")

# Check finetuned model is in the current directory if chosing the finetuned model option
def check_model_folder():
    folder_name = 'fine-tuned-model'
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, folder_name)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        st.sidebar.write(f'{model_bart} model selected')
    else:
        st.sidebar.write(f"Please download and unzip the finetuned model into current working directory \n https://drive.google.com/file/d/1X7HAtapky6u9HZq1-nDQu7tQ0kXJ3F3F/view?usp=share_link")

# Load Whisper model for vidoe transcribing
def load_whisper_model(model_name):
    model = whisper.load_model(model_name)
    return model

# Summarise the given transcript with either pretrained or fintuned model
def summarize_text(text, option='Pretrained', progress_bar=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if option == 'Pretrained':
        transformer = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    elif option == 'Finetuned':
        transformer = BartForConditionalGeneration.from_pretrained('fine-tuned-model').to(device)
        tokenizer = BartTokenizer.from_pretrained('fine-tuned-model')

    # Break the transcrpt into smaller chunck to process for reduced memory requirement
    chunk_size = 1024
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Generate summary for each chunks and concatenate the summaries
    summaries = []
    for i, chunk in enumerate(chunks):
        inputs = tokenizer.encode(chunk, return_tensors='pt').to(device)
        summary_ids = transformer.generate(inputs, max_length=50, min_length=10, do_sample=True, temperature=0.7, top_k=50)[0]
        summary = tokenizer.decode(summary_ids, skip_special_tokens=True)
        summaries.append(summary)

        # Update progress bar
        if progress_bar is not None:
            progress_bar.progress((i+1) / len(chunks))

    final_summary = ' '.join(summaries)

    return final_summary

st.set_page_config(page_title="Video Summariser", page_icon=":movie_camera:")
st.header("Video Summariser")

file = st.file_uploader("Select a video file", type=["mp4", "webm"])
if file is not None:
    save_file(file, file.name)
    video_bytes = file.read()
    st.video(video_bytes)

options = {'tiny': 'Fastest Option', 'base': 'Balanced Option', 'small': 'Most Accurate Option'}
model_name = st.sidebar.radio('Select a Whisper Model:', list(options.keys()), format_func=lambda option: f"{option}: {options[option]}")
model = whisper.load_model(model_name)
st.sidebar.write(f'{model_name} model selected')

options_bart = {'Pretrained': 'BART pretrained from CNN News', 'Finetuned': 'Pretrained model finetuned on TED talks'}
model_bart = st.sidebar.radio('Select a BART Model:', list(options_bart.keys()), format_func=lambda option: f"{option}: {options_bart[option]}")
if model_bart == 'Finetuned':
    check_model_folder()
else:
    st.sidebar.write(f'{model_bart} model selected')

if st.sidebar.button('Summarise Video'):
    if file is not None:
        with st.spinner('Transcribing Video...'):
            transcript = model.transcribe(file.name)
        st.sidebar.success('Transcription Complete')
        st.sidebar.success('Summarising Video')
        progress_bar_sum = st.sidebar.progress(0)
        result = summarize_text(transcript['text'], model_bart, progress_bar_sum)
        progress_bar_sum.empty()
        st.sidebar.success('Summarisation Complete')
        st.markdown(result)
    else:
        st.sidebar.error("Please upload file of correct format")
