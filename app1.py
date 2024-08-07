from flask import Flask, request, jsonify
import subprocess
import os
import inference
MODEL = inference.Predictor()

app = Flask(__name__)

# Function to download files using wget
def download_file(url, path):
    try:
        subprocess.run(['wget', '-O', path, url], check=True, text=True, capture_output=True)
        print(f'Successfully downloaded {url} to {path}')
    except subprocess.CalledProcessError as e:
        print(f'Error downloading {url} to {path}: {e.stderr}')

@app.route('/run-inference', methods=['POST'])
def run_inference():
    # Get the data in the JSON request
    data = request.json
    source_image_https = data.get('source_image')
    driving_video_https = data.get('driving_video')

    # Download the source image and driving video
    source_image = os.path.basename(source_image_https)
    driving_video = os.path.basename(driving_video_https)
    source_image_path = os.path.join('assets/users', source_image)
    driving_video_path = os.path.join('assets/users', driving_video)
    download_file(source_image_https, source_image_path)
    download_file(driving_video_https, driving_video_path)

    if not source_image or not driving_video:
        return jsonify({"error": "Missing source_image or driving_video parameter"}), 400

    # 构建命令
    # command = f'python inference.py -s {source_image_path} -d {driving_video_path}'

    # 运行命令
    try:
        # subprocess.run(command, check=True, shell=True)
        output = MODEL.predict(source=source_image_path, driving=driving_video_path)
        return jsonify({"message": output}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
