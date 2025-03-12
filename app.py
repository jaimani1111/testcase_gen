import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st
from docx import Document
from io import BytesIO

#  env
load_dotenv()
print("Environment variables loaded successfully.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("Groq API key loaded.")


groq_client = Groq(api_key=GROQ_API_KEY)
print("Groq client initialized successfully.")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("HuggingFace embeddings initialized for Chroma DB compatibility.")

# Load test bank
test_bank = pd.read_csv("test.csv")
print("Test bank loaded successfully.")
print(f"Test bank columns: {test_bank.columns}")
print(f"Number of rows in test bank: {len(test_bank)}")

# Load Chroma DB with precomputed embeddings
try:
    print("Loading Chroma DB with precomputed embeddings...")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    print("Chroma DB loaded successfully.")
except Exception as e:
    print(f"Error loading Chroma DB: {e}")
    st.error(f"Error loading Chroma DB: {e}")
    st.stop()

# Function to clean description
def clean_description(description):
    # Remove unwanted characters like *
    return description.replace("*", "").strip()

# func to generate new test cases using Groq
def generate_test_case(insurance_type, region, line_of_business, user_requirements):
    # Retrieve similar test cases from chromaDB
    query = f"Insurance Type: {insurance_type}, Line of Business: {line_of_business}, User Requirements: {user_requirements}"
    similar_cases = vector_db.similarity_search(query, k=3)
    context = "\n".join([case.page_content for case in similar_cases])
    
    # Generating  new test case using Groq , passing prompt  template
    prompt = f"""
   Generate a completely new test case for Insurance Type: {insurance_type}, 
Region: {region}, generate specific for that region from your understanding
Line of Business: {line_of_business}, 
User Requirements: {user_requirements}. 

The test case must be completely different from the retrieved test cases in the context and cover edge cases such as invalid inputs, extreme values, or unusual scenarios. Ensure that the test case is not similar to the test cases already provided in the  {context} but in same format.

Use the following format:
- Sl No.
- Requirement ID
- Test Case ID
- Module
- LOB
- Region
- Test Case Description
- Execution Steps
- Expected Result

Context from similar test cases: {context}
    """
    print(f"Query sent to Groq: {prompt}")
    
    # Use Groq client to generate a response
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
    )
    print("Response received from Groq.")
    return response.choices[0].message.content

# Streamlit App
st.markdown("<h1 style='text-align: center;'>XC AI Assisted Test Cases Generator</h1>", unsafe_allow_html=True)

# Inputs
with st.sidebar:
    st.header("Select Inputs")
    insurance_type = st.selectbox("Which insurance type do you want to select?", ["Select", "Health", "Auto", "Home", "Life"])
    region = st.selectbox("Which region do you want to select?", ["Select", "North America", "Europe", "Asia Pacific", "ANZ"])
    line_of_business = st.selectbox("Which Line of Business do you want to select?", ["Select", "Retail", "Commercial", "Enterprise"])

    # Upload Acceptance Criteria File
    st.subheader("Upload Acceptance Criteria File")
    criteria_file = st.file_uploader("Drop a file here or browse", type=["txt", "csv", "docx"])
    st.caption("Limit 200MB per file • TXT, CSV, DOCX")

    # Func extract text 
    def extract_criteria(file):
        if file is None:
            return ""
        if file.type == "text/plain":
            return file.getvalue().decode("utf-8")
        if file.type == "text/csv":
            df = pd.read_csv(file)
            return "\n".join(df.iloc[:, 0].dropna().astype(str))  # Extract first column
        if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            return "\n".join([p.text for p in doc.paragraphs])
        return ""

    # Extract criteria e
    uploaded_criteria = extract_criteria(criteria_file)

    # Acceptance Criteria Input - Disabled if a file is uploaded
    st.subheader("Enter Acceptance Criteria (or upload a file)")
    if criteria_file:
        st.text_area("Acceptance Criteria", uploaded_criteria, height=100, disabled=True)
    else:
        manual_criteria = st.text_area("Write Acceptance Criteria", "", height=100)

    # Generate Test Case Button
    if st.button("Generate Test Case", key="generate_button"):
        st.session_state.generate_clicked = True
    else:
        st.session_state.generate_clicked = False

# Use uploaded criteria if available, otherwise use manual input
final_criteria = uploaded_criteria if uploaded_criteria else manual_criteria

# Display test cases in the center AFTER generation
if final_criteria and st.session_state.generate_clicked:
    st.success("Acceptance Criteria Ready.")
    
    st.subheader("Generated Test Cases")
    
    # Generate test cases using Groq
    new_test_case = generate_test_case(insurance_type, region, line_of_business, final_criteria)
    print("New test case generated.")
    st.write("### Generated Test Case")
    st.write(new_test_case)

    # Clean the description
    cleaned_description = clean_description(new_test_case)

    # Parse the generated test case into the required format
    new_test_case_data = {
        "Sl No.": len(test_bank) + 1,  # Auto-increment serial number
        "Requirement ID": "REQ_NEW",  # Placeholder for requirement ID
        "Test Case ID": f"TC_{len(test_bank) + 1}",  # Auto-increment test case ID
        "Module": line_of_business,
        "LOB": insurance_type,
        "Region": region,
        "Test Case Description": cleaned_description,
        "Execution Steps": "Step 1: Navigate to the module.\nStep 2: Enter details.\nStep 3: Submit and verify.",  # Placeholder steps
        "Expected Result": "Successfully processed.",  # Placeholder expected result
    }
    print("New test case data parsed successfully.")
    print(f"New test case data: {new_test_case_data}")

    # Append the new test case to the test bank
    new_test_case_df = pd.DataFrame([new_test_case_data])
    updated_test_bank = pd.concat([test_bank, new_test_case_df], ignore_index=True)
    print("New test case appended to the test bank.")

    # Save the updated test bank to CSV
    updated_test_bank.to_csv("test.csv", index=False)
    print("Test bank saved to CSV successfully.")
    st.success("Test case saved to CSV successfully.")

    # Download Section
    st.subheader("Download Test Cases")
    file_format = st.radio("Select file format:", ["CSV", "Word"])

    def convert_to_csv(test_case_data):
        df = pd.DataFrame([test_case_data])
        # Format Execution Steps with line breaks for better readability in Excel
        df["Execution Steps"] = df["Execution Steps"].str.replace("\n", "\\n")
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return output

    def convert_to_word(test_case_data):
        doc = Document()
        doc.add_heading("Generated Test Case", level=1)
        doc.add_paragraph(f"Sl No.: {test_case_data['Sl No.']}")
        doc.add_paragraph(f"Requirement ID: {test_case_data['Requirement ID']}")
        doc.add_paragraph(f"Test Case ID: {test_case_data['Test Case ID']}")
        doc.add_paragraph(f"Module: {test_case_data['Module']}")
        doc.add_paragraph(f"LOB: {test_case_data['LOB']}")
        doc.add_paragraph(f"Region: {test_case_data['Region']}")
        doc.add_paragraph(f"Test Case Description: {test_case_data['Test Case Description']}")
        doc.add_paragraph(f"Execution Steps: {test_case_data['Execution Steps']}")
        doc.add_paragraph(f"Expected Result: {test_case_data['Expected Result']}")
        output = BytesIO()
        doc.save(output)
        output.seek(0)
        return output

    if file_format == "CSV":
        st.download_button("Download CSV", convert_to_csv(new_test_case_data), "test_case.csv", mime="text/csv")
    elif file_format == "Word":
        st.download_button("Download Word", convert_to_word(new_test_case_data), "test_case.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")