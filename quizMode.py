import openai
import streamlit as st
import chardet
import pdfplumber

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

def read_file(uploaded_file):
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    else:
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] if result['encoding'] is not None else 'utf-8'
        return raw_data.decode(encoding, errors='replace')

def generate_questions_from_text(text, num_true_false, num_mcq, num_essay):
    prompt = (
        "Generate a set of quiz questions based on the following content. "
        "Include the specified number of true/false, multiple choice, and essay questions:\n\n"
        f"{text}\n\n"
        "The number of questions should be:\n"
        f"- True/False: {num_true_false}\n"
        f"- Multiple Choice: {num_mcq}\n"
        f"- Essay: {num_essay}\n\n"
        "Provide the questions and answers in plain text format. "
        "Explain why the answer is correct for each question"
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # You can adjust this depending on the model you're using
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates quiz questions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.7,
        n=1,
        stop=None
    )

    return response['choices'][0]['message']['content'].strip()

def main():
    st.title("File-Based Quiz Generator")

    uploaded_file = st.file_uploader("Upload a file", type=["pdf"])
    
    if uploaded_file is not None:
        # Input fields for the number of different types of questions
        num_true_false = st.number_input("Number of True/False Questions", min_value=0, value=1, step=1)
        num_mcq = st.number_input("Number of Multiple Choice Questions", min_value=0, value=1, step=1)
        num_essay = st.number_input("Number of Essay Questions", min_value=0, value=1, step=1)

        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                content = read_file(uploaded_file)
                questions_text = generate_questions_from_text(content, num_true_false, num_mcq, num_essay)
            
            # Display all generated questions and answers in text format
            st.header("Generated Quiz Questions")
            st.write(questions_text)

if __name__ == "__main__":
    main()

