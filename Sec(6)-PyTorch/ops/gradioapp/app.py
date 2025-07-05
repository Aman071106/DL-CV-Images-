import gradio as gr
from core.predict import ImageClassifier
import os
from PIL import Image

# Load model once when app starts
class_dict = {0: 'Cat', 1: 'Dog', 2: 'Person'}
classifier = ImageClassifier(model_path=None, class_dict=class_dict)

# Inference function
def predict(image):
    uploaded_img_path = "uploaded.jpg"
    image.save(uploaded_img_path)

    label, output_path = classifier.predict(uploaded_img_path)

    return label, Image.open(output_path)

# Gradio app interface
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Image(label="Labeled Image")
    ],
    title="Image Classification Gradio App",
    description="Upload an image to classify it as Dog, Cat, or Person"
)

# Launch the app
if __name__ == "__main__":
    app.launch()
