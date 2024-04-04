import streamlit as st
import pandas as pd

st.set_page_config(
        page_title="MRI",
        page_icon="robot"
)

# Define CSS style for the heading
heading_style = (
    "text-align: center;"
    "font-size: 30px;"
    "font-weight: bold;"
    "padding: 20px 0;"
)

# Display the heading with the defined style
st.markdown("<h1 style='{}'>Icon Idea '24 - Preventive & Predictive Maintenance Magnetic Resonance Imaging (MRI)</h1>".format(heading_style), unsafe_allow_html=True)
 
st.subheader("Solution Overview:")
st.write("""
**Preventive Maintenance:** Preventive Maintenance is a regular and routine maintenance of an equipment in order to maintain functioning in an expected behavior manner to meet the purpose and avoid unplanned downtime from malfunctions.

**Predictive Maintenance:** Predictive Maintenance is to keep track the health of an equipment/system and derive the timelines when the equipment can undergo the required calibration.
""")

st.subheader("Objective:")
st.write("""
The objective of preventive maintenance is to avoid equipment failure by predicting its occurrence, and to reduce the risk of unplanned downtime and accidental damages of the equipment.
1. Correlate all health parameters of entire system components
2. The model is developed on past history data of an equipment (Gen AI based on machine learning), create / fine tune the current model with relevant past data and validate with real time data that collected from respective equipment.
""")

image = "images\image001.png"
st.image(image, use_column_width=True)

image = "images\image002.png"
st.image(image, use_column_width=True)

st.subheader("Current System vs Proposed System:")
data = {
    'Current Solution': [
        'Maintenance of equipment are prescheduled',
        'Medical Equipment and its components health are being monitored and measured independently',
        'Identifying and isolating the root cause is post incident activity'
    ],
    'Your solution': [
        'Based on past history, data collected and current performance., Gen AI based system can determine when the equipment can undergo the preventive maintenance tasks.',
        'Medical equipment health parameters are being monitored, measured with the allowed ranges recommended by OEM and / or history data and correlated among them to determine the actual performance of whole system and identify the medical equipment or component faults early before the system fails.',
        'As this system keep track of all required performance indicators / health parameters, measured with the allowed ranges and highlighted the anomalies, it helps the supporting staff to identify and isolating the root causes as and when the incident is occurred.'
    ]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)
st.write(df)


st.subheader("Benefits:")
st.write("""
a) Increase the medical equipment lifespan,\n
b) Lower risk of breakdowns.\n
c) Increase service efficiency\n
d) Eliminates / reduces the unplanned downtime associated with medical equipment malfunctions;\n
e) Promote health and safety/n
f) Boost patient safety & hygiene.\n
g) Save money & time\n
h) Saves natural resources i.e. energy / water by analyzing consumption data and optimizing usage.
""")

st.subheader("Preventive & Predictive Maintenance Tasks:")
st.write("""
1. Physical / Visual Inspection
    - Eyeball inspection
    - Wear and tear
    - Cable cuts, cracks in cable insulations
    - Ambience temperature and environmental variables, noise levels, door close
    - Medical physicist
2. Radio Frequency (RF) transmitter gain / Attenuator verification
    - Monitor RF coil  using phantom scan sequence
    - Lot of coils not designed uniform, have characteristic hot spots and cold spots
3. Systems Performance - KPI Data
4. Signal to Noise ratio (SNR) evaluation
5. Slice thickness evaluation
6. Slice Accuracy evaluation
7. Image Intensity Uniformity
8. Artifact Assessment
""")

st.subheader("Annual System Assessments:")
st.write("""
1. Magnet field homogeneity assessment
2. Slice thickness accuracy assessment
3. Slice position accuracy assessment
4. High contrast Spatial resolution assessment
5. System artifact assessment
6. Low Contrast detectability / contrast to noise-ratio assessment
7. General equipment assessment
""")

image = "images\image003.png"
st.image(image, use_column_width=True)

image = "images\image004.png"
st.image(image, use_column_width=True)

image = "images\image005.png"
st.image(image, use_column_width=True)

image = "images\image006.png"
st.image(image, use_column_width=True)

st.subheader("Solution - Software & Tools:")
st.write("""
    1. OS: Windows Operating system 10
    2. LLM: Google Gemini Pro
    3. IDE: VS Code v1.87.2
    4. Python: v3.10.13
""")
         
st.subheader("Python Libraries:")
st.write("""
    1. streamlit: v1.32.2
    2. langchain:  v0.1.13
    3. langchain-google-genai: v0.0.11
    4. google-generativeai: v0.4.1
    5. python-dotenv: v1.0.1
""")

st.header("Future Steps:")
st.write("""
    1. Develop LLM to support with Operations Manual of Siemens Healthineers Magnetom family MRI Systems (1.5 tesla) through Retrieval Augmented Generation (RAG) for more precise recommendations and step by step approach to avoid unplanned downtimes.
    2. Develop a Chat bot for Radiology Supported engineers to get detailed steps on executing the recommendations while troubleshooting the reported issues
    3. UI changes - to overcome Streamlit limitations (probably replace with Fast API)
    4. 1 year (365 days) of Historical data which covers all seasonal and cyclic impacts on MRI's Performance and maintenances carried out to train the LLM
    5. Add additional number of Key performance Indicators and relevant supported KPIs (IAQ, Chiller cooling and Ambient environment parameters)
    6. Optimize the Classifier model approach to analyze more data sets and patterns on history data and correlate more number of KPIs
    7. Dynamic approach for Prompt engineering - create prompt, context and question dynamically based on the realtime data
    8. Authentication & Authorization
    9. Information Security enablement
""")

st.header("Team Behind:")
image = "images\image007.png"
st.image(image, use_column_width=True)


