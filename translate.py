import re
import streamlit as st
import sentencepiece as spm
import ctranslate2
from nltk import sent_tokenize
from sacremoses import MosesTokenizer, MosesDetokenizer
mt = MosesTokenizer(lang='en')
import fugashi
import neologdn
import truecase
import s3fs
import os
import subprocess

# Title for the page and nice icon
st.set_page_config(page_title="NMT", page_icon="🤖")
# Header
st.title("jp-translate.io")

@st.cache
def download_Unidic():
    returned_value = subprocess.run(sys.executable, "-m", "unidic", "download")  # returns the exit code in unix
    print(returned_value)

download_Unidic()

# Create AWS S3 connection object.
fs = s3fs.S3FileSystem(anon=False)

# Initialize tokenizers
mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')
tagger = fugashi.Tagger('-Owakati')

@st.cache
def build_directories():
    #First construct ENJP directory, then JPEN
    files = fs.ls('ctranslate2models/ENJP_ctranslate2/')
    #Make a staging directory that can hold data as a medium
    if not os.path.exists("ENJP_ctranslate2"):
        os.mkdir("ENJP_ctranslate2")

    with st.spinner("Downloading models... this will only take a few seconds! \n Don't stop it!"):
        for file in files:
            item = str(file)
            lst = item.split("/")
            name = lst[2]
            path = "ENJP_ctranslate2\\" + name
            fs.download(file, path)

    files = fs.ls('ctranslate2models/JPEN_ctranslate2/')
    if not os.path.exists("JPEN_ctranslate2"):
        os.mkdir("JPEN_ctranslate2")
        
    with st.spinner("Downloading models... this will only take a few seconds! \n Don't stop it!"):
        for file in files:
            item = str(file)
            lst = item.split("/")
            name = lst[2]
            path = "JPEN_ctranslate2\\" + name
            fs.download(file, path)

def translate(source, translator, sp_source_model, sp_target_model):
    """Use CTranslate model to translate a sentence

    Args:
        source (str): Source sentences to translate
        translator (object): Object of Translator, with the CTranslate2 model
        sp_source_model (object): Object of SentencePieceProcessor, with the SentencePiece source model
        sp_target_model (object): Object of SentencePieceProcessor, with the SentencePiece target model
    Returns:
        Translation of the source text
    """

    if option == "English-to-Japanese":
        source = source.lower()
        source_sentences = sent_tokenize(source)
        source_tokenized = sp_source_model.encode(source_sentences, out_type=str)
        translations = translator.translate_batch(source_tokenized)
        translations = [translation[0]["tokens"] for translation in translations]
        translations_detokenized = sp_target_model.decode(translations)
        translation = " ".join(translations_detokenized)
        normalized = neologdn.normalize(translation)
        return normalized

    if option == "Japanese-to-English":
        source = re.split(r'(?<=\。)', source)
        newlist = []
        for sentence in source:
            source_sentences = tagger.parse(sentence)
            newlist.append(source_sentences)
        source_tokenized = sp_source_model.encode(newlist, out_type=str)
        translations = translator.translate_batch(source_tokenized)
        translations = [translation[0]["tokens"] for translation in translations]
        translations_detokenized = sp_target_model.decode(translations)
        mosesdetok = md.detokenize(translations_detokenized)
        truecased = truecase.get_true_case(mosesdetok)
        return truecased

def load_models(option):

    if option == "English-to-Japanese":
        ct_model_path = "ENJP_ctranslate2"
        sp_source_model_path = "EN_Final.model"
        sp_target_model_path = "JP_Final.model"

    elif option == "Japanese-to-English":
        ct_model_path = "JPEN_ctranslate2"
        sp_source_model_path = "JP_Final.model"
        sp_target_model_path = "EN_Final.model"

    # Create objects of CTranslate2 Translator and SentencePieceProcessor to load the models
    translator = ctranslate2.Translator(ct_model_path, "cpu")    # or "cuda" for GPU
    sp_source_model = spm.SentencePieceProcessor(sp_source_model_path)
    sp_target_model = spm.SentencePieceProcessor(sp_target_model_path)

    return translator, sp_source_model, sp_target_model

build_directories()

# Form to add your items
with st.form("my_form"):

    # Dropdown menu to select a language pair
    option = st.selectbox(
    "Select Language Pair",
    ("English-to-Japanese", "Japanese-to-English"))
    #st.write('You selected:', option)

    # Textarea to type the source text.
    user_input = st.text_area("Source Text", max_chars=2000)

    # Load models
    translator, sp_source_model, sp_target_model = load_models(option)
    
    # Translate with CTranslate2 model
    translation = translate(user_input, translator, sp_source_model, sp_target_model)

    # Create a button
    submitted = st.form_submit_button("Translate")
    # If the button pressed, print the translation
    # Here, we use "st.info", but you can try "st.write", "st.code", or "st.success".
    if submitted:
        st.write("Translation")
        st.info(translation)

st.markdown('Interested in how this was built? Read the research paper :book:') 

with open('Research.pdf',"rb") as f:
   st.download_button('Download', f, "Research.pdf")

st.markdown('Interested in improving this project? [Contact me](https://matthewbieda.github.io/)') 
