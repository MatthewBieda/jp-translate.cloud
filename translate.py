import re
from click import option
import streamlit as st
import sentencepiece as spm
import ctranslate2
from nltk import sent_tokenize
from sacremoses import MosesTokenizer, MosesDetokenizer
mt = MosesTokenizer(lang='en')
import fugashi
import neologdn
import truecase

# Title for the page and nice icon
st.set_page_config(page_title="jp-translate.io", page_icon="random")
# Header
st.title("jp-translate.io")

# Initialize tokenizers
mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')
tagger = fugashi.Tagger('-Owakati')

@st.cache(allow_output_mutation=True)
def load_model():
        
    if option == "English-to-Japanese":
        ct_model_path = "ENJP_ctranslate2"
        sp_source_model_path = "EN_Final.model"
        sp_target_model_path = "JP_Final.model"

    if option == "Japanese-to-English":
        ct_model_path = "JPEN_ctranslate2"
        sp_source_model_path = "JP_Final.model"
        sp_target_model_path = "EN_Final.model"

    # Create objects of CTranslate2 Translator and SentencePieceProcessor to load the models
    translator = ctranslate2.Translator(ct_model_path, "cpu")    # or "cuda" for GPU
    sp_source_model = spm.SentencePieceProcessor(sp_source_model_path)
    sp_target_model = spm.SentencePieceProcessor(sp_target_model_path)

    return translator, sp_source_model, sp_target_model

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
        newarray = []
        splitsource = source.splitlines()
        for entry in splitsource:
            source_sentences = sent_tokenize(entry)
            source_tokenized = sp_source_model.encode(source_sentences, out_type=str)
            translations = [translation[0]["tokens"] for translation in translations]
            translations_detokenized = sp_target_model.decode(translations)
            translation = " ".join(translations_detokenized)
            normalized = neologdn.normalize(translation)
            newarray.append(normalized)
        final = "\n".join(newarray)
        return final

    if option == "Japanese-to-English":
        # use regex to split input sentences on fullstop (keeping it) into a list
        source = re.split(r'(?<=\。)', source)
        newlist = []
        # optimal to use mecab to parse sentence by sentence
        for sentence in source:
            source_sentences = tagger.parse(sentence)
            sourceaslist = [source_sentences]
            source_tokenized = sp_source_model.encode(sourceaslist, out_type=str)
            translations = translator.translate_batch(source_tokenized)
            translations = [translation[0]["tokens"] for translation in translations]
            translations_detokenized = sp_target_model.decode(translations)
            mosesdetok = md.detokenize(translations_detokenized)
            truecased = truecase.get_true_case(mosesdetok)
            if '\n' in sentence:
                newlist.append('\n' + truecased)
            else:
                newlist.append(truecased)
        final = "\n".join(newlist)
        return final

# Form to add your items
with st.form("my_form"):

    # Dropdown menu to select a language pair
    option = st.selectbox(
    "Select Language Pair",
    ("English-to-Japanese", "Japanese-to-English"))

    # Textarea to type the source text.
    user_input = st.text_area("Source Text", max_chars=2000)

    # Load models
    translator, sp_source_model, sp_target_model = load_model()
    
    # Translate with CTranslate2 model
    translation = translate(user_input, translator, sp_source_model, sp_target_model)

    # Create a button
    submitted = st.form_submit_button("Translate")
    # If the button pressed, print the translation
    # Here, we use "st.info", but you can try "st.write", "st.code", or "st.success".
    if submitted:
        st.write("Translation")
        st.info(translation)

st.markdown('Interested in how this was built? [Read the research paper](https://arxiv.org/abs/2202.11669) :book:') 

st.markdown('Interested in improving this project? [Contact me](https://matthewbieda.github.io/)')

st.text(" ")

st.markdown("<p style='text-align: center;'>Created by Matthew Bieda. Assisted by Yasmin Moslem.</p>", unsafe_allow_html=True)
 

# Optional Style (courtesy of ymoslem)
st.markdown(""" <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .reportview-container .main .block-container{
        padding-top: 0rem;
        padding-right: 0rem;
        padding-left: 0rem;
        padding-bottom: 0rem;
    } </style> """, unsafe_allow_html=True)
