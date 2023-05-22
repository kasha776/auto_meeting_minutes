# %%
# 匯入必要的模組
import app as st
import pandas as pd
import numpy as np
from pydub import AudioSegment
import math
from tqdm import tqdm
import os
from dotenv import load_dotenv
import streamlit as st
import openai
import io

# 創建主頁面
st.title('會議摘要自動生成器')
# %%
#parameter settings

load_dotenv()

openai.api_key = os.getenv("api_key")

# %%
def split_audio_by_minute(audio):
    # audio = AudioSegment.from_file(audio_file)
    duration = len(audio)
    minute_length = 60 * 1000  # 分鐘長度（以毫秒為單位）

    num_segments = math.ceil(duration / minute_length)  # 分鐘數

    for i in range(num_segments):
        start_time = i * minute_length
        end_time = min((i + 1) * minute_length, duration)  # 最後一個片段可能不足一分鐘
        segment = audio[start_time:end_time]

        segment.export('temp_audio/mp3_segment_{}.mp3'.format(i), format="mp3")
        
def save_string_to_txt(string, file_path):
    with open(file_path, 'w') as file:
        file.write(string)

def get_completion(prompt, model="gpt-3.5-turbo",temperature=0): # Andrew mentioned that the prompt/ completion paradigm is preferable for this class
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def wrap_text(text, max_width = 35):
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line) + len(word) <= max_width:
            current_line += word + " "
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    
    # 添加最後一行
    lines.append(current_line.strip())
    
    return "\n".join(lines)
# %%

# 創建一個側邊攔的選單
options = ['逐字稿','音檔']
choice = st.selectbox('選擇檔案類型', options)

# 首頁
if choice == '音檔':
    # 上傳檔案
    uploaded_file = st.file_uploader(f"選擇一個{choice}檔案", type=['mp4','m4a','mp3'])
    if st.button("Run meeting"):
        transcript_all = ''
        if uploaded_file is not None:
            # 將上傳的檔案轉換為二進位制檔案物件
            file_object = io.BytesIO(uploaded_file.read())

            # 根據檔案類型使用 AudioSegment 來讀取音檔
            if uploaded_file.type == "audio/mpeg":
                audio = AudioSegment.from_mp3(file_object)
            elif uploaded_file.type == "audio/mp4" or uploaded_file.type == "audio/x-m4a":
                audio = AudioSegment.from_file(file_object, format="mp4")
            else:
                st.warning("不支援的檔案類型")
        # 使用示例
            segments = split_audio_by_minute(audio)

        folder_path = "temp_audio"  # 替换为你的文件夹路径

        file_list = os.listdir(folder_path)
        file_count = len(file_list) -1

        progress_bar = st.progress(0)
        for i in tqdm(range(file_count)):
            # 在這裡對音訊片段做進一步的處理
            # 例如，可以對每個片段進行特定的音訊處理或分析
            audio_file = open('temp_audio/mp3_segment_{}.mp3'.format(i), "rb")
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            transcript_all = transcript_all + transcript.text
            progress_value = (i+1)/file_count
            progress_bar.progress(progress_value)
        st.empty()
        # col1, col2 = st.columns([1, 2])
        # with col1:
        st.sidebar.header("逐字稿內容")
        st.sidebar.text(wrap_text(transcript_all).replace('\n', '  \n'))

        start_idx = 200
        result = '' 
        overlap_token = 200 

        # create a progress bar
        progress_bar = st.progress(0)

        total_length = len(transcript_all)
        while start_idx < total_length:
            start_idx = start_idx - overlap_token
            end_idx = min(start_idx + 2500, total_length)
            sub_list = transcript_all[start_idx:end_idx]
            prompt = f"你現在是一個會議紀錄專業寫手，\
            你的任務是根據會議逐字稿整理會議紀錄，\
            會議紀錄需要包含每一項討論主題、討論內容與對應的下一步，\
            其中討論主題需要明確，可以再討論主題下拆分不同的子主題\
            討論內容需要在40個文字內讓讀者可以理解，\
            下一步行動則需要有具體的人需要執行哪些事情，若沒有明確下一步行動則顯示為無。\
            文字內容會以'''呈現，\
            請直接輸出優化後的結果\
            不要有空白\
            文字內容：'''{sub_list}'''"
            response = get_completion(prompt)
            result += response
            start_idx = end_idx

            # update the progress bar
            progress_value = start_idx / total_length
            progress_bar.progress(progress_value)

        prompt = f"你現在是一個 json 格式達人，\
        請你先閱讀完全部的檔案內容之後，再將格式調整為 csv\
        並且限制使用 1900 個以下的字串 數完整輸出內容，\
        json 檔案內容：'''{result}'''"

        resonse_final = get_completion(prompt)
        stringio = io.StringIO(resonse_final)
        data = pd.read_csv(stringio, sep = ',')
        # with col2:
        st.header("會議摘要")
        st.dataframe(data)
elif choice == '逐字稿':
    uploaded_file = st.file_uploader(f"選擇一個{choice}檔案", type=['txt'])
    if st.button("Run meeting"):
        transcript_all = uploaded_file.read().decode("utf-8")
        st.empty()
        # col1, col2 = st.columns([1, 2])
        # with col1:
        st.sidebar.header("逐字稿內容")
        st.sidebar.text(wrap_text(transcript_all).replace('\n', '  \n'))
        start_idx = 200
        result = '' 
        overlap_token = 200 

        # create a progress bar
        progress_bar = st.progress(0)

        total_length = len(transcript_all)
        while start_idx < total_length:
            start_idx = start_idx - overlap_token
            end_idx = min(start_idx + 2500, total_length)
            sub_list = transcript_all[start_idx:end_idx]
            prompt = f"你現在是一個會議紀錄專業寫手，\
            你的任務是根據會議逐字稿整理會議紀錄，\
            會議紀錄需要包含每一項討論主題、討論內容與對應的下一步，\
            其中討論主題需要明確，可以再討論主題下拆分不同的子主題\
            討論內容需要在40個文字內讓讀者可以理解，\
            下一步行動則需要有具體的人需要執行哪些事情，若沒有明確下一步行動則顯示為無。\
            文字內容會以'''呈現，\
            請直接輸出優化後的結果\
            不要有空白\
            文字內容：'''{sub_list}'''"
            response = get_completion(prompt)
            result += response
            start_idx = end_idx

            # update the progress bar
            progress_value = start_idx / total_length
            progress_bar.progress(progress_value)

        prompt = f"你現在是一個 json 格式達人，\
        請你先閱讀完全部的檔案內容之後，再將格式調整為 csv\
        json 檔案內容：'''{result}'''"

        resonse_final = get_completion(prompt)
        stringio = io.StringIO(resonse_final)
        data = pd.read_csv(stringio, sep = ',')
        # with col2:
        st.header("會議摘要")
        st.dataframe(data)

# %%



