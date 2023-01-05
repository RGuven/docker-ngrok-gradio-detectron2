import os,shutil
import gradio as gr

from utils.formatter import clips_formatter
from utils.segmentation import Segmentation


UPLOAD_FOLDER_PATH = ".//_uploads"
RESULT_FOLDER_PATH=".//_results"


#Load The Segmentation Model 
segmentation_service = Segmentation()

def segmentation(img_temp_file_path):
    image_file_name = os.path.basename(img_temp_file_path)
    
    #new image path which is located in the _uploads folder
    image_file_path = f"{UPLOAD_FOLDER_PATH}//{image_file_name}"

    # copy the original image from temp folder to _uploads folder.
    shutil.copy(img_temp_file_path, image_file_path)

    # give the saved image to model for prediction
    response_data = segmentation_service.prediction(image_file_name)


    prediction={
        "clips":clips_formatter(response_data)
    }

    predicted_image = f"{RESULT_FOLDER_PATH}//{image_file_name}"
    return predicted_image,prediction
    

    

def main():    
    examples = [['gradio_ui_example_images/1122815383.jpg'], ['gradio_ui_example_images/1122815389.jpg'],
                ['gradio_ui_example_images/1122815390.jpg'], ['gradio_ui_example_images/1122815391.jpg'],
                ['gradio_ui_example_images/1122927103.jpg']]


    title= "Newspaper Layout Segmentation Project"

    demo = gr.Interface(segmentation, 
                        gr.Image(type="filepath",label="Newspaper Page").style(height=700), 
                        [gr.Image().style(height=700),"json"],
                        examples=examples,
                        title=title)
                        
    demo.launch(server_name="0.0.0.0",server_port=8888)


if __name__ == '__main__':
    main()