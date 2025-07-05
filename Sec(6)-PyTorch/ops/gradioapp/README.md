---
title: CNN Classifier
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
license: mit
---

# CNN_classifier
A cnn  based project for image classification


### How to deploy a gradio app on hugging face space ?
- >pip install "huggingface_hub[cli]"
- >huggingface-cli login
- enter personal access token

Then you can deploy  to hugginface with this project structure in two ways:
1. hugging face repo [method](https://www.udemy.com/course/complete-computer-vision-bootcamp-with-pytoch-tensorflow/learn/lecture/49094327#overview)
This method is actually more stable and works in a go.
2. git repo linked to hugging face from doc that is used here

[Colab Notebook](https://colab.research.google.com/drive/16BU4aL_mGSxUBQHmyK9BO5xYjJ5aB2M8)

* Before space deployment , deploy your model in new model section in hugging face

### Some keypoints while deployment
- Gradio has bugged versions so  use the latest one as specified in requirements.txt
and also for client
- Don't push binary files like images and files >10 mb on space
- Even if pushed use 