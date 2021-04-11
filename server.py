from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import urllib.request as urllib
from enum import Enum
import traceback
import requests
import uvicorn
import base64
import json
import uuid
import sys
import ssl
import os
import re

from inference import init_detector, inference_detector, draw_results_on_image

model1 = "ResNet50"
model2 = "ResNet101"

# pair{weight_path:config_file}
models_params = {model1: ["Custom-Model-res50.pth", "r50_config.py"],
                 model2: ["Custom-Model-res101.pth", "r101_config.py"]}

models = {model1: None, model2: None}

app = FastAPI()

# The root is the absolute path of the __init_.py under the source
ROOT = os.path.abspath(__file__)[:os.path.abspath(__file__).rfind(os.path.sep)]
ROOT_DOWNLOAD_URL = './.data_cache'

url_downloader = urllib.FancyURLopener(
    context=ssl._create_unverified_context())

app = FastAPI(title="Custom Model Inference")


# init_model(model_name) # init all models


class InputModel(BaseModel):
    back_url: str = None
    threshold: float = 0.55
    model_name: str
    image_file: str


class model_name(str, Enum):
    ResNet50 = model1
    ResNet101 = model2


def init_model(model_name: str):
    print("Calling function init_model")
    if models[model_name] is not None:
        return

    fCheckpoint = "path to checkpoint file .pth"
    fConfig = "path to model config or model.py itself"
    if not os.path.exists(fCheckpoint) or not os.path.exists(fConfig):
        raise RuntimeError(
            "directory: 'checkpoints' and/or 'configs' does not exist!")

    models[model_name] = init_detector(fConfig, fCheckpoint, device='cuda:0')
    return


def run_detector(input_model: InputModel, input_image_file: str):
    print("calling function run_detector")
    return "inference complete"
    model = models[input_model.model_name]

    result = inference_detector(model, input_image_file)
    name, ext = os.path.split(input_image_file)[1].split(".")
    name += "Gen."
    name += ext

    out_img = os.path.join(os.path.dirname(input_image_file), name)
    draw_results_on_image(input_image_file, result, out_file=out_img)
    return out_img


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
        super(InferenceProcessTask, self).__init__()
        self.func = func
        self.input_data = input_data
        self.response_data = dict()

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
                    ROOT_DOWNLOAD_URL, self.image_cache_id)
                url_downloader.retrieve(
                    self.input_data.image_file, input_image_file)
            except Exception as e:
                print(e)
                self.response_data["code"] = "failed"
                self.response_data['msg'] = "Can not download image from \'%s\'. Not a valid link." % (
                    self.input_data.image_file)
                return

        # Prepare the results
        # run the inference function
        self.results = self.func(self.input_data, input_image_file)
        self.response_data["code"] = "success"
        # iterate through results
        # TODO remove when result format is confirmed
        """
        for res in self.results:
            if res.status == "Failure":
                self.response_data["code"] = "failed"
                self.response_data['msg'] = "Failed to process image"
                break

            self.response_data['msg'] = "Processed image successfully"
            self.response_data["num_detections"] = len(self.results)
            resp_dict_item = {}
            with open(input_image_file, mode='rb') as file:
                img = file.read()
            resp_dict_item["File"] = base64.b64encode(img)
            # base64.b64decode(Iod_item["File"]) # for decoding
            self.response_data["results"] = resp_dict_item
        """
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
        except Exception as e:
            print(e)


@app.post("/inference_model_file/{inputModel}_model")
async def inference_model_file(input_model: model_name,
                               background_tasks: BackgroundTasks,
                               file: UploadFile = File(...),
                               display_image_only: bool = Form(True),
                               threshold: float = Form(0.5)):
    response_data = dict()
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
            back_url=None,
            threshold=threshold)
        task = InferenceProcessTask(run_detector,
                                    input_data=input_data)
        task.run()
        response_data = task.response_data
        if display_image_only:
            return FileResponse(path=image_file_path,
                                media_type=file.content_type)
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        response_data["code"] = "failed"
        response_data["msg"] = "failed to run inference on file"

    return response_data


@app.post("/inference_model_url/{inputModel}_model")
async def inference_model_url(*, input_model: model_name,
                              input_data: InputModel,
                              background_tasks: BackgroundTasks):
    response_data = dict()
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

    except Exception as e:
        print(e)
        response_data["code"] = "failed"
        response_data["msg"] = "failed to run inference on file from url"

    return response_data


@app.get("/")
def index():
    return {"Welcome to Model Inference Server Web Service": "Please visit /docs"}


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # if port is not specified
        print("Using default port: 8080")
        uvicorn.run(app, host='0.0.0.0', port=8080)
    elif len(sys.argv) == 2:
        # port specified
        print("Using port: " + sys.argv[1])
        uvicorn.run(app, host='0.0.0.0', port=int(sys.argv[1]))
