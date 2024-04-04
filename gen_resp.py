import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.environ["GEMINI_API_KEY"]

# Initialize GenerativeAI model
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=api_key)

prompt = ChatPromptTemplate.from_template("""
Providing MRI Scan data of 1.5 Tesla MRI System. The label names are as follows
scan_type,scan_time,snr (dB),drift (Hz),drift_ppm(ppm),grad_perf(G/cm/ms),coil_type,error_temp (°C),sys_temp (°C),cyro_boiloff(liters/hour),rf_power (%),grad_temp (°C),grad_current (A),x_axis_pos (mm),y_axis_pos (mm),z_axis_pos (mm),preds and corresponding values are given below
Data: {data}

Context: error 111 is related to shortened scan time due to software glitch in specific scan sequence, error 102 is related to Excessive drift in magnet subsystem, error 7 is related to System temperature exceeding safe limit during high-power sequence, error 201 is related to Low signal-to-noise ratio in cardiac images likely due to malfunctioning coil element and error 304 is related to Reduced gradient current during rapid angiography sequence might indicate, error 402 is related to Injector malfunction - contrast not administered properly, error 501 is related to Motion artifact due to patient breathing during long scan, error 603 is related to RF transmission error - signal not reaching tissues properly, error 707 is related to Flow artifact during angiography sequence obscuring blood vessels, and error 809 is related to Image quality degradation due to excessive patient motion

Question: Analyze the provided values of the MRI Scan data and suggest actions to be performed on MRI to avoid reoccurrence of this error.
""")

output_parser = StrOutputParser()

chain = prompt | model | output_parser


def generate_response(data):
    try:
       # Generate response
        response = chain.invoke({"data": data})
    except:
        response = "Sorry, I couldn't generate a response."
    return response
