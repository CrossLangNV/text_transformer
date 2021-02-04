import os
import json
import subprocess

import flask

app = flask.Flask(__name__)

TMP_FILE_FOR_INPUT = 'input.txt'
TMP_FILE_FOR_OUTPUT = 'output.txt'


@app.route('/')
def home():
    return 'COMPRISE Text Transformer Service'


@app.route('/transform', methods=['POST'])
def transform_text():
    params = json.loads(flask.request.args.items().__next__()[0])

    with open(TMP_FILE_FOR_INPUT, mode='w') as f:
        text = flask.request.data.decode()
        f.write(text)

    cmd = ["python", "transform.py"]
    for key, value in params.items():
        cmd.extend(['-' + key, str(value)])
    cmd.extend([os.path.abspath(TMP_FILE_FOR_INPUT), os.path.abspath(TMP_FILE_FOR_OUTPUT)])
    subprocess.run(cmd, cwd='./transformer')

    with open(TMP_FILE_FOR_OUTPUT, mode='r') as f:
        result = ''.join(f.readlines())
    return result


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
