from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/run-inference', methods=['POST'])
def run_inference():
    # Get the data in the JSON request
    data = request.json
    source_image = data.get('source_image')
    driving_video = data.get('driving_video')
    if not source_image or not driving_video:
        return jsonify({"error": "Missing source_image or driving_video parameter"}), 400

    # 构建命令
    command = f'python inference.py -s {source_image} -d {driving_video}'

    # 运行命令
    try:
        subprocess.run(command, check=True, shell=True)
        return jsonify({"message": "Inference completed successfully"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
