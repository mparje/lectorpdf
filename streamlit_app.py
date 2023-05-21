import streamlit as st
import PyPDF2
from io import BytesIO
import torch
from transformers import LongformerTokenizer, LongformerForQuestionAnswering

def search_and_highlight_result(text, search_term):
    highlighted_text = text.replace(search_term, f"<mark>{search_term}</mark>")
    return highlighted_text

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-base-4096")

def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt', truncation='only_second', max_length=4096)
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = tokenizer.convert_tokens_to_string(tokens[answer_start:answer_end+1])
    
    return answer

st.set_page_config(page_title='Guatemalan Law Search', layout='wide', initial_sidebar_state="expanded")

st.title('Guatemalan Law Search App')

uploaded_file = st.sidebar.file_uploader("Upload a PDF File", type=["pdf"])
search_query = st.sidebar.text_input("Search for a term or phrase or ask a question")

user_page_number = st.sidebar.number_input("Jump to page (type the page number)", min_value=1, value=1, step=1)

if uploaded_file is not None:
    pdfReader = PyPDF2.PdfFileReader(uploaded_file)
    total_pages = len(pdfReader.pages)

    pdf_extracted_text = ""
    for page_index in range(total_pages):
        page = pdfReader.pages[page_index]
        page_content = page.extractText()
        pdf_extracted_text += page_content

    st.sidebar.write(f"Total pages: {total_pages}")

    if search_query != "":
        answer = answer_question(search_query, pdf_extracted_text)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.subheader("Document Contents:")
        st.markdown(pdf_extracted_text, unsafe_allow_html=True)
