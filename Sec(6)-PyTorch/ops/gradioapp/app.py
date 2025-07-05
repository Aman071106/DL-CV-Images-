import gradio as gr
from core.predict import ImageClassifier
import os
from PIL import Image


model_path=os.path.join(os.getcwd(),'model','cnn_model.pth')
class_dict = {0: 'Cat', 1: 'Dog', 2: 'person'}
classifier=ImageClassifier(model_path=model_path,class_dict=class_dict)

# inference frontend
def predict(image):
    uploaded_img_path='uploaded.jpg'
    image.save(uploaded_img_path)
    
    label,output_path=classifier.predict(uploaded_img_path)
    
    return label,Image.open(output_path)

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=[gr.Textbox(label="Prediction"), gr.Image(label="Labeled Image")],
    title="Image Clascification Gradio app",
    description= "Upload an image to classify it as Dog, Cat,or Person"
)

if __name__ == "__main__":
    app.launch()