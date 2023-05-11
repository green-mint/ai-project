from flask import Flask, request, send_from_directory
import uuid
from ocr.ocr import load_model, load_characters, predict_image

characters = load_characters('ocr/chars.txt')
model = load_model('ocr/urdu.model', len(characters), device='cpu').to('cpu')
model.eval()

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload():
    print("req received")
    if 'image' not in request.files:
        return 'No image file found', 400

    image = request.files['image']
    # fname should be a uuid + the image name + the extension
    fname = f'{uuid.uuid4()}{image.filename}'

    # generate a uuid and write the file
    image.save(f'./images/{fname}')

    return f'./images/{fname}.jpg', 200


@app.route('/predict', methods=['POST'])
def predict():
    # if fname or dims not in body return 400
    data = request.get_json()
    # if 'fname' or 'x' or 'y' or 'w' or 'h' not in data:
    #     return 'fname or dims not in body', 400

    # get fname and dims from body
    fname = data['fname']
    dims = (data['x'], data['y'], data['w'], data['h'])
    # print(fname, dims)
    # predict the image
    label = predict_image(model, characters, f'./images/{fname}', dims)
    print(label)
    # return the label
    return label, 200


@app.route('/images/<path:filename>')
def get_image(filename):
    return send_from_directory('images', filename)


if __name__ == '__main__':
    app.run(debug=True)
