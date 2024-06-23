from flask import Flask, request, render_template, url_for
import os
import sys
from werkzeug.utils import secure_filename

# 외부 스크립트 폴더 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'stable-diffusion-pytorch'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'CartoonGAN-Test-Pytorch-Torch'))
from create import create_img
from convert import convert_image

app = Flask(__name__)

# 업로드 폴더 설정
GENERATED_FOLDER = 'static/generated_images'
CONVERTED_FOLDER = 'static/converted_images'
UPLOADED_FOLDER = 'static/uploaded_images'

app.config['GENERATED_FOLDER'] = GENERATED_FOLDER
app.config['CONVERTED_FOLDER'] = CONVERTED_FOLDER
app.config['UPLOADED_FOLDER'] = UPLOADED_FOLDER

def return_filepath(path, name):
    replaced_name = name.replace(" ", "_")
    filename = secure_filename(f"{replaced_name}.png")  # 텍스트의 앞 10자를 파일 이름으로 사용
    return os.path.join(path, filename)

def exist_path(filepath):
    print(filepath)
    print(os.path.exists(filepath))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    prompt = request.form['prompt']
    img = request.files['img']
    n_inference_steps = request.form['inferencesteps']
    sampler = request.form['sampler']

    img_path = return_filepath(UPLOADED_FOLDER, img.filename)
    img.save(img_path)

    filepath = return_filepath(GENERATED_FOLDER, prompt)
    print(exist_path(filepath))

    image = create_img(prompt, img_path, int(n_inference_steps), sampler)
    image.save(filepath, 'PNG')

    return render_template('result.html', prompt=prompt, filepath=filepath)


@app.route('/convert', methods=['POST'])
def convert():
    img = request.files['img1']

    img_path = return_filepath(UPLOADED_FOLDER, img.filename)
    img.save(img_path)

    styles = ['Shinkai', 'Paprika', 'Hosoda', 'Hayao']

    output_paths = []

    for style in styles:
        output_path = return_filepath(CONVERTED_FOLDER, img.filename + '_' + style)
        convert_image(img_path, style, output_path)

        output_paths.append(output_path)

    return render_template('convert.html',
                           style1_txt=styles[0], image1_url=output_paths[0],
                           style2_txt=styles[1], image2_url=output_paths[1],
                           style3_txt=styles[2], image3_url=output_paths[2],
                           style4_txt=styles[3], image4_url=output_paths[3]
           )

if __name__ == '__main__':
    app.run(debug=True)
