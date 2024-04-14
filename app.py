from flask import Flask, jsonify, request
from flask_cors import CORS 
import pandas as pd
from mri_dt import mri_classifier_model
from gen_resp import generate_response
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


app = Flask(__name__)
CORS(app)


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Initialize GenerativeAI model
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=api_key)

df = pd.read_csv("mri_qa.csv")



@app.route('/mri_maintenance/<int:row_number>', methods=['GET'])
def get_mri_data(row_number):
    mri_classifier_model()
    data = []
    with open('output.csv', 'r') as f:
        next(f) 
        for l in f:
            line = l.strip().split(',')
            data.append(line)

    if row_number >= 1 and row_number <= len(data):
        line = data[row_number - 1]
        error_code = line[-1]
        scan_details = {
            'sl.no': line[1],
            'scan_type': line[2],
            'scan_time': line[3],
            'snr': line[4],
            'drift': line[5],
            'drift_ppm': line[6],
            'grad_perf': line[7],
            'coil_type': line[8],
            'error_temp': line[9],
            'sys_temp': line[10],
            'cyro_boiloff': line[11],
            'rf_power': line[12],
            'grad_temp': line[13],
            'grad_current': line[14],
            'x_axis_pos': line[15],
            'y_axis_pos': line[16],
            'z_axis_pos': line[17],
            'error_code': error_code
        }

        if error_code != "No Error":
            llm_output = generate_response(line)
            scan_details['recommended_actions'] = llm_output
        else:
            scan_details['recommended_actions'] = "Scan is successful, MRI performance is normal"

        return jsonify(scan_details)
    else:
        return jsonify({'error': 'Invalid row number'}), 400
    

@app.route('/quality_assessment', methods=['GET'])
def quality_assessment():
    df = pd.read_csv("mri_qa.csv")

    # Prompt template to query GenerativeAI
    prompt = ChatPromptTemplate.from_template("""
    MRI Quality Assessment - tests are performed 8 times on each assessment and below given are the results.

    The first line contains headers of data and below data is in csv format.

    Test Name,Test Type,1st Result,2nd Result,3rd Result,4th Result,5th Result,6th Result,7th Result,8th Result

    Geometric accuracy (mm),Localizer,-2,0,-1,-2,-2,1,-2,1

    Geometric accuracy (mm),Slice 1 mm with vertical dimension,-2,-1,-2,0,0,0,0,0

    Geometric accuracy (mm),Slice 1 mm with horizontal dimension,-1,-1,-1,-1,0,-1,0,0

    Geometric accuracy (mm),Slice 5mm with vertical dimension,-2,-1,-2,-2,0,0,0,-1

    Geometric accuracy (mm),Slice 5mm with horizontal dimension,-1,0,-2,0,-1,-1,-1,-1

    Geometric accuracy (mm),Slice 5mm with diagonal dimension with upper left and lower right,-3,-1,-1,0,-1,0,-1,0

    Geometric accuracy (mm),Slice 5mm with diagonal dimension with upper right and lower left,-3,0,-1,-1,0,0,0,0

    High-contrast spatial resolution (mm),Test 1 with horizontal resolution,1,1,1,1,1,1,1,1

    High-contrast spatial resolution (mm),Test 1 with vertical resolution,1,1,1,1,1,1,1,1

    High-contrast spatial resolution (mm),Test 2 with horizontal resolution,1,1,1,1,1,1,1,1

    High-contrast spatial resolution (mm),Test 2 with vertical resolution,1,1,1,1,1,1,1,1

    Slice thickness accuracy (mm),Test 1,0.3,0.6,0.5,0.4,1.4,1.2,0.4,0.6

    Slice thickness accuracy (mm),Test 2,0.2,0.1,0.2,0,0.3,0.4,-0.1,0.1

    Slice position accuracy (mm),Test 1 with bias in superior slice,1,1.5,1,-2,1.5,1.5,0,-1

    Slice position accuracy (mm),Test 1 with bias in inferior slice,2,1,1,-1,0.5,0,0.5,-0.5

    Slice position accuracy (mm),Test 2 with bias in superior slice,1.5,1,1.5,-1.5,2,1.5,0,-1

    Slice position accuracy (mm),Test 2 with bias in inferior slice,2.3,2,1,-1,0.5,0,0.5,-0.5

    Image intensity Uniformity (%),Test 1,88,91,93,92,91,93,92,92

    Image intensity Uniformity (%),Test 2,89,93,92,94,91,92,92,94

    Percent signal ghosting (%),Test 1,0.04,0.8,0.05,0.03,1.2,0.4,0.2,0.4

    Low-contrast object detectability (contrast level),Test 1 with 1.4 percent of contrast level,0,3,9,8,9,8,9,5

    Low-contrast object detectability (contrast level),Test 1 with 2.5 percent of contrast level,9,9,10,9,10,9,10,6

    Low-contrast object detectability (contrast level),Test 1 with 3.6 percent  of contrast level,9,10,10,9,10,10,10,9

    Low-contrast object detectability (contrast level),Test 1 with 5.1 percent  of contrast level,9,10,10,10,10,10,10,10

    Low-contrast object detectability (contrast level),Test 2 with 1.4 percent of contrast level,0,2,7,2,8,4,8,5

    Low-contrast object detectability (contrast level),Test 2 with 2.5 percent of contrast level,0,8,10,8,9,9,9,7

    Low-contrast object detectability (contrast level),Test 2 with 3.6 percent  of contrast level,2,9,10,9,9,9,10,9

    Low-contrast object detectability (contrast level),Test 2 with 5.1 percent  of contrast level,9,10,10,9,9,10,10,10

    Signal-to-noise ratio,Test 1,283,427,343,294,164,139,382,206

    Signal-to-noise ratio,Test 2,181,284,227,158,144,83,283,121

    Central frequency (Hz),Test 1,63870647,63870545,63867509,63869411,63590094,63589389,40481340,40481282

    

    Consider 'Geometric accuracy (mm)' allowed range is +/-1 and 0, 'High-contrast spatial resolution (mm)' allowed range is 1, 'Slice thickness accuracy (mm)' allowed range should be less than 1, 'Slice position accuracy (mm)' allowed range is +/-1 and 0, 'Image intensity Uniformity (%)' allowed range is greater than or equal to 90%, 'Percent signal ghosting (%)' should be less than 1, 'Low-contrast object detectability (contrast level)' allowed range is greater than 7

    

    Based on data given above, analyze result data of each test with above given allowed ranges and share the output of each test in given format.

    'Name' should be Test Name,

    'Observations' can be analyzed the result data of each test with respective Test allowed ranges and provide count of fails and count of success with comma separated and show the '% success rate'; highlight '% success rate' in bold letters.

    'Actions' based on observation with result data of each test with respective Test allowed ranges, identify the faults and improvements, suggest precise actions in professional way to avoid downtime of MRI system,

    'Maintenance' specify which component maintenance is required of MRI System; and highlight them in Bold letters.

    'Urgency (Approx.)' as "Immediate", "in a week", "in a month", "in 3 months" of time.

    The output should use the below labels; Output should not be crossed 4000 tokens.

    Name:

    Observations:

    Actions: 

    Maintenance:

    Urgency (Approx.):

    The output should be shown in tabular format with the above column.""")

    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    question = "Based on data given above, what actions can be performed to avoid downtime of MRI and suggest actions for predictive maintenance on components and how soon we need to perform? Analyze the data first, before conclusion. provide the actions, suggestions at each result level."

    context = """Geometric accuracy (mm) allowed range is +/-1 and 0,High-contrast spatial resolution (mm) allowed range is 1,Slice thickness accuracy (mm) allowed range should be less than 1,Slice position accuracy (mm) allowed range is +/-1 and 0,Image intensity Uniformity (%) allowed range is greater than or equal to 90%,Percent signal ghosting (%) should be less than 1,Low-contrast object detectability (contrast level) allowed range is greater than 7"""

    def generate_response():
        try:
            response = chain.invoke({})
        except:
            response = "Sorry, I couldn't generate a response."
        return response

    output = generate_response()

    scan_details = {
        'title': 'MRI Quality Assessment',
        'description': 'MRI Annual Quality Assessment Data collected from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3047180/',
        'data': df.to_dict('records'),
        'observations_actions_recommendations': output
    }

    return jsonify(scan_details)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@app.route('/mri_assistant', methods=['POST'])
def ask_question():
    try:
        user_question = request.form.get('question')
        pdf_files = request.files.getlist('pdf_files')

        if not user_question:
            return jsonify({'error': 'Question is required'}), 400

        if not pdf_files:
            return jsonify({'error': 'PDF files are required'}), 400

        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return jsonify({'response': response["output_text"]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)