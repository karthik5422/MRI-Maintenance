import time
import pandas as pd
import streamlit as st
from mri_dt import mri_classifier_model
from gen_resp import generate_response

st.markdown("<h1 style='text-align: center;'>Magnet Resonance Imaging (MRI)</h1>", unsafe_allow_html=True)
st.subheader("Current Status & Actions for Maintenance: Based on Scaned Data")

mri_classifier_model()

with open('output.csv', 'r') as f :
    next(f)
    for l in f:
        line = l.strip().split(',')
        st.write("**Current scan details with MRI performence data**")
        df = pd.DataFrame([line], columns=['sl.no','scan_type','scan_time','snr (dB)','drift (Hz)','drift_ppm(ppm)','grad_perf(G/cm/ms)','coil_type','error_temp (°C)','sys_temp (°C)','cyro_boiloff(liters/hour)','rf_power (%)','grad_temp (°C)','grad_current (A)','x_axis_pos (mm)','y_axis_pos (mm)','z_axis_pos (mm)','Error_Code'])
        st.table(df)
        st.write("**Current Status:**")
        
        error_code = line[-1]
        if error_code != "No Error":
            llm_output = generate_response(line)
            st.write("**There is an error found while scaning. The below actions are recommended.**")
            st.write(llm_output)
        else:
            st.write("**Scan is successful, MRI performance is normal**")
        st.divider()

        time.sleep(1)
