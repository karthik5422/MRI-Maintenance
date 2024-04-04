import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

api_key = os.environ["GEMINI_API_KEY"]

# Initialize GenerativeAI model
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=api_key)

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

'Urgency' as "Immediate", "in a week", "in a month", "in 3 months" of time.

The output should use the below labels; Output should not be crossed 4000 tokens.

Name:

Observations:

Actions: 

Maintenance:

Urgency:

The output should be shown in tabular format with the above column.""")


output_parser = StrOutputParser()

chain = prompt | model | output_parser

question = "Based on data given above, what actions can be performed to avoid downtime of MRI and suggest actions for predictive maintenance on components and how soon we need to perform? Analyze the data first, before conclusion.  provide the actions, suggestions at each result level."

context = """Geometric accuracy (mm) allowed range is +/-1 and 0,High-contrast spatial resolution (mm) allowed range is 1,Slice thickness accuracy (mm) allowed range should be less than 1,Slice position accuracy (mm) allowed range is +/-1 and 0,Image intensity Uniformity (%)   allowed range is greater than or equal to 90%,Percent signal ghosting (%) should be less than 1,Low-contrast object detectability (contrast level) allowed range is greater than 7"""

def generate_response():
    try:
       # Generate response
        response = chain.invoke({})
    except:
        response = "Sorry, I couldn't generate a response."
    return response

output=generate_response()

st.markdown("<h1 style='text-align: center;'>MRI Quality Assessment</h1>", unsafe_allow_html=True)
st.write("MRI Annual Quality Assessmnet Data collected from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3047180/")
st.write("9 Different tests are performed 8 times each, the below table contains the results.")
st.write(df)
st.write("**Observations, Actions and Recommendations**")
st.write(output)