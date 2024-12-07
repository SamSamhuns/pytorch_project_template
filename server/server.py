import urllib.request as urllib2
from enum import Enum
import traceback
import json
import uuid
import sys
import os
import re

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import uvicorn
from inference_server import init_detectors, run_detector


# The root is the absolute path of the __init_.py under the source
ROOT = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind(os.path.sep)]
ROOT_DOWNLOAD_URL = os.path.join(ROOT, ".data_cache")

app = FastAPI(title="Custom Model Inference")

# load models here
model1 = "mnist_model_1"
model2 = "mnist_model_11"
# pair{weight_path:config_file}
models_param_dict = {model1: ["checkpoints_server/models/best.pth", "checkpoints_server/models/config.json"],
                     model2: ["checkpoints_server/models/best.pth", "checkpoints_server/models/config.json"]}
# init all models
loaded_models_dict = init_detectors(models_param_dict, device="cpu")


class InputModel(BaseModel):
    back_url: str = None
    threshold: float = 0.55
    model_name: str
    image_file: str


class model_name(str, Enum):
    ResNet50 = model1
    ResNet101 = model2


def download_url_file(download_url, download_path) -> None:
    """Download file from download_url to download_path
    """
    response = urllib2.urlopen(download_url)
    with open(download_path, 'wb') as f:
        f.write(response.read())


def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


async def cache_file(name: str, data: bytes):
    print("Caching Image")
    image_file_path = os.path.join(ROOT_DOWNLOAD_URL, name)
    os.makedirs(ROOT_DOWNLOAD_URL, exist_ok=True)
    with open(image_file_path, 'wb') as img_file_ptr:
        img_file_ptr.write(data)
    return image_file_path


class InferenceProcessTask():
    def __init__(self, func, input_data):
        super().__init__()
        self.func = func
        self.input_data = input_data
        self.response_data = {}

    def run(self):
        # check if the input image_file is an existing path or a url
        is_local_dir = os.path.exists(self.input_data.image_file)
        is_url = re.match(r'^https?:/{2}\w.+$', self.input_data.image_file) \
            or re.match(r'^https?:/{2}\w.+$', self.input_data.image_file)

        if is_local_dir:
            input_image_file = self.input_data.image_file
        elif is_url:
            try:
                os.makedirs(ROOT_DOWNLOAD_URL, exist_ok=True)
                input_image_file = os.path.join(
                    ROOT_DOWNLOAD_URL, str(uuid.uuid4()) + '.jpg')
                download_url_file(self.input_data.image_file, input_image_file)
            except Exception as excep:
                print(excep)
                self.response_data["code"] = "failed"
                self.response_data['msg'] = "Can not download image from \'%s\'. Not a valid link." % (
                    self.input_data.image_file)
                return

        # Prepare the results
        # run the inference function
        inference_model = loaded_models_dict[self.input_data.model_name]
        self.result = self.func(
            inference_model, input_image_file, self.input_data.threshold)
        self.response_data["code"] = "success"
        self.response_data["prediction"] = self.result
        # iterate through results
        # remove when result format is confirmed
        # """
        # for res in self.results:
        #     if res.status == "Failure":
        #         self.response_data["code"] = "failed"
        #         self.response_data['msg'] = "Failed to process image"
        #         break
        #
        #     self.response_data['msg'] = "Processed image successfully"
        #     self.response_data["num_detections"] = len(self.results)
        #     resp_dict_item = {}
        #     with open(input_image_file, mode='rb') as file:
        #         img = file.read()
        #     resp_dict_item["File"] = base64.b64encode(img)
        #     # base64.b64decode(Iod_item["File"]) # for decoding
        #     self.response_data["results"] = resp_dict_item
        # """
        # Remove cached file
        if is_url:
            if os.path.exists(input_image_file):
                os.remove(input_image_file)
        try:
            if self.input_data.back_url is not None:
                headers = {"Content-Type": "application/json"}
                requests.request(method="POST",
                                 url=self.input_data.back_url,
                                 headers=headers,
                                 data=json.dumps(self.response_data),
                                 timeout=(3, 100))
                print("successfully sent")
        except Exception as excep:
            print(excep)


@app.post("/inference_model_file/{inputModel}_model")
async def inference_model_file(input_model: model_name,
                               background_tasks: BackgroundTasks,
                               file: UploadFile = File(...),
                               display_image_only: bool = Form(False),
                               threshold: float = Form(0.55)):
    response_data = {}
    image_file_path = ""
    try:
        # Save this image to the temp file
        file_name = str(uuid.uuid4()) + '.jpg'
        file_bytes_content = file.file.read()
        image_file_path = await cache_file(file_name, file_bytes_content)
        # add cached file removal to list of bg tasks to exec after sending response
        background_tasks.add_task(remove_file, image_file_path)

        input_data = InputModel(
            model_name=input_model.value,
            image_file=image_file_path,
            back_url="",
            threshold=threshold)
        task = InferenceProcessTask(run_detector,
                                    input_data=input_data)
        task.run()
        response_data = task.response_data
        if display_image_only:
            return FileResponse(path=image_file_path,
                                media_type=file.content_type)
    except Exception as excep:
        print(excep)
        print(traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = "failed to run inference on file"

    return response_data


@app.post("/inference_model_url/{inputModel}_model")
async def inference_model_url(*, input_model: model_name,
                              input_data: InputModel,
                              background_tasks: BackgroundTasks):
    response_data = {}
    try:
        task = InferenceProcessTask(
            run_detector,
            input_data=input_data)

        if input_data.back_url is None:
            task.run()
            # if the user doesn't provide returning url, we just wait
        else:
            background_tasks.add_task(task.run)
        response_data = task.response_data

    except Exception as excep:
        print(excep)
        print(traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = "failed to run inference on file from url"

    return response_data


@app.get("/")
def index():
    return {"Welcome to Model Inference Server Web Service": "Please visit /docs"}


if __name__ == '__main__':
    if len(sys.argv) == 1:    # if port is not specified
        print("Using default port: 8080")
        uvicorn.run(app, host='0.0.0.0', port=8080)
    elif len(sys.argv) == 2:  # port specified
        print("Using port: " + sys.argv[1])
        uvicorn.run(app, host='0.0.0.0', port=int(sys.argv[1]))
