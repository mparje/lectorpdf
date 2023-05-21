import streamlit as st
import PyPDF2
from io import BytesIO
import openai
import os

def search_and_highlight_result(text, search_term):
    highlighted_text = text.replace(search_term, f"<mark>{search_term}</mark>")
    return highlighted_text

def answer_question_davinci(question, context):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{question}\n\n{context}",
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )

    answer = response.choices[0].text.strip()
    return answer

st.set_page_config(page_title='Guatemalan Law Search', layout='wide', initial_sidebar_state="expanded")

st.title('Guatemalan Law Search App')

uploaded_file = st.sidebar.file_uploader("Upload a PDF File", type=["pdf"])
search_query = st.sidebar.text_input("Search for a term or phrase or ask a question")

user_page_number = st.sidebar.number_input("Jump to page (type the page number)", min_value=1, value=1, step=1)

if uploaded_file is not None:
    pdfReader = PyPDF2.PdfReader(uploaded_file)
    total_pages = len(pdfReader.pages)

    pdf_extracted_text = ""
    for page_index in range(total_pages):
        page = pdfReader.pages[page_index]
        page_content = page.extract_text()
        pdf_extracted_text += page_content

    st.sidebar.write(f"Total pages: {total_pages}")

    if search_query != "":
        answer = answer_question_davinci(search_query, pdf_extracted_text)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.subheader("Document Contents:")
        st.markdown(pdf_extracted_text, unsafe_allow_html=True)
