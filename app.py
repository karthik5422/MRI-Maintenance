from flask import Flask, jsonify
from flask_cors import CORS 
import pandas as pd
from mri_dt import mri_classifier_model
from gen_resp import generate_response

app = Flask(__name__)
CORS(app)

@app.route('/mri_data/<int:row_number>', methods=['GET'])
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
            'sl.no': row_number,
            'scan_type': line[1],
            'scan_time': line[2],
            'snr': line[3],
            'drift': line[4],
            'drift_ppm': line[5],
            'grad_perf': line[6],
            'coil_type': line[7],
            'error_temp': line[8],
            'sys_temp': line[9],
            'cyro_boiloff': line[10],
            'rf_power': line[11],
            'grad_temp': line[12],
            'grad_current': line[13],
            'x_axis_pos': line[14],
            'y_axis_pos': line[15],
            'z_axis_pos': line[16],
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

if __name__ == '__main__':
    app.run(debug=True)