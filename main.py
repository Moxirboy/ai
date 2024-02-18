from flask import Flask, request, jsonify , render_template
from torchvision import transforms
from PIL import Image
import torch
from skin_model import AdvancedSkinCNN  
import io 
import generate  as gn
import translate as trans

app = Flask(__name__)

# Initialize the model
model = AdvancedSkinCNN()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'response': 'Please provide an image file'}), 400
    file = request.files['file']
    content_type = file.content_type
    print("Received file content type:", content_type)


    img_bytes = file.read()
    class_id = predict_image(img_bytes)
    # Example mapping, update with your actual classes
    conditions = {
        0: 'No illness detected',
        1: 'Melanoma',
        2: 'Benign keratosis',
        3: 'Basal cell carcinoma',
        4: 'Actinic keratosis',
        5: 'Vascular lesion',
        6: 'Dermatofibroma',
    }
    condition = conditions.get(class_id, 'Unknown condition')
    print(condition)
    wa_id = "1"
    name = "moxirboy"
    response = gn.generate_response(condition, wa_id, name)
    # response=trans.translate_text(new_message, "en", "uz",)
    print(response)
    return jsonify({"response": response})

@app.route('/generate_response', methods=['POST'])
def generate_response_api():
    data = request.get_json()
    wa_id = "1"
    name = "moxirboy"
    message_body = data.get('request')
    # message=trans.translate_text(message_body, "uz", "en",)
    response = gn.generate_response(message_body, wa_id, name)
    # response=trans.translate_text(new_message, "en", "uz",)
    return jsonify({"response": response})
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
