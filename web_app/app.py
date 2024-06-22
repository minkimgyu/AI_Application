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
UPLOAD_FOLDER = 'static/generated_images'
CONVERTED_FOLDER = 'static/converted_images'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(CONVERTED_FOLDER):
    os.makedirs(CONVERTED_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CONVERTED_FOLDER'] = CONVERTED_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    user_text = request.form['user_text']
    image = create_img(user_text)

    # 파일 이름 생성 및 저장
    filename = secure_filename(f"{user_text}.png")  # 텍스트의 앞 10자를 파일 이름으로 사용
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath, 'PNG')

    image_url = url_for('static', filename=f'generated_images/{filename}')
    return render_template('result.html', user_text=user_text, image_url=image_url)


@app.route('/convert', methods=['POST'])
def convert():
    user_text = request.form['user_text']

    print(user_text)
    file = user_text.replace(" ", "_")
    print(file)

    style = 'Shinkai'

    filename = secure_filename(f"{file}.png")  # 텍스트의 앞 10자를 파일 이름으로 사용

    # 파일 변환
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)


    # 파일 저장
    output_filename = secure_filename(f"{file}_{style}.jpg")  # 텍스트의 앞 10자를 파일 이름으로 사용
    output_path = url_for('static', filename=f'converted_images')

    # 파일을 저장하고 경로만 반환해준다.
    convert_image(filepath, output_filename, style, output_path)

    return render_template('convert.html', user_text=user_text, image_url=output_path + '/' + output_filename)

if __name__ == '__main__':
    app.run(debug=True)
