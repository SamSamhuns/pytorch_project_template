from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import urllib.request as urllib
from enum import Enum
import requests
import base64
import uvicorn
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
ROOT_DOWNLOAD_URL = './data_cache'

url_downloader = urllib.FancyURLopener(
    context=ssl._create_unverified_context())

app = FastAPI(title="Custom Model Inference")


class InputModel(BaseModel):
    back_url: str = None
    threshold: float = 0.55
    image_file: str


class model_name(str, Enum):
    ResNet50 = model1
    ResNet101 = model2


def make_path(fileName: str, where: str):
    print("calling function make_path")
    root_dir = os.path.abspath(os.path.curdir)

    if where == "chkpt":
        d = os.path.join(root_dir, "checkpoints")

    if where == "confg":
        d = os.path.join(root_dir, "configs/CustomModel")

    return os.path.join(d, fileName)


def init_model(model_name: str):
    print("Calling function init_model")
    if models[model_name] is not None:
        return

    fCheckpoint = make_path(models_params[model_name][0], "chkpt")
    fConfig = make_path(models_params[model_name][1], "confg")
    if not os.path.exists(fCheckpoint) or not os.path.exists(fConfig):
        raise RuntimeError(
            "directory: 'checkpoints' and/or 'configs' does not exist!")

    models[model_name] = init_detector(fConfig, fCheckpoint, device='cuda:0')
    return


def run_detector(model_name: str, input_img: str):
    # image stored in cache folder
    print("calling function run_detector")
    print("key = ", model_name)
    init_model(model_name)

    model = models[model_name]

    result = inference_detector(model, input_img)
    name, ext = os.path.split(input_img)[1].split(".")
    name += "Gen."
    name += ext

    out_img = os.path.join(os.path.dirname(input_img), name)
    draw_results_on_image(input_img, result, model.CLASSES, out_file=out_img)
    return out_img


class InferenceProcessTask():
    def __init__(self, func, input_data, threshold=0.55):
        super(InferenceProcessTask, self).__init__()
        self.func = func
        self.input_data = input_data
        self.threshold = threshold
        self.response_data = dict()
        self.input_data_content_type = None
        self.image_cache_id = str(uuid.uuid4())
        self.response_data["uuid"] = self.image_cache_id

    def run(self):
        # check if the input image_file is an existing path or a url
        is_local_dir = os.path.exists(self.input_data.image_file)
        is_url = re.match(r'^https?:/{2}\w.+$', self.input_data.image_file) \
            or re.match(r'^https?:/{2}\w.+$', self.input_data.image_file)

        if is_local_dir:
            input_image_file = self.input_data.image_file
            threshold = self.threshold
        elif is_url:
            try:
                if not os.path.exists(ROOT_DOWNLOAD_URL):
                    os.mkdir(ROOT_DOWNLOAD_URL)
                input_image_file = os.path.join(
                    ROOT_DOWNLOAD_URL, self.image_cache_id)
                url_downloader.retrieve(
                    self.input_data.image_file, input_image_file)
                threshold = self.threshold
            except Exception as e:
                print(e)
                self.response_data["code"] = "failed"
                self.response_data['msg'] = "Can not download image from \'%s\'. Not a valid link." % (
                    self.input_data.image_file)
                return

        # Prepare the results
        # run the inference function
        self.results = self.func(input_image_file, threshold)
        self.response_data["code"] = "success"
        # iterate through results
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


@app.post("/inference_model/{inputModel}_file/")
async def process_img(inputModel: model_name,
                      file: UploadFile = File(...)):
    print("Accessing route " + inputModel.value)
    name = file.filename
    content = file.content_type
    if "image" not in content:
        return {"ERROR": "file content is not an image"}

    data = await file.read()
    await file.close()
    imgPath = await cache_image(name, data)
    outImg = run_detector(inputModel.value, imgPath)
    return FileResponse(path=outImg, media_type=content)


@app.post("/inference_model_file")
async def img_object_detection_file(file: UploadFile = File(...),
                                    display_image_only: bool = Form(True),
                                    threshold: float = Form(0.5)):
    response_data = dict()
    try:
        # Save this image to the temp file
        file_name = str(uuid.uuid4()) + '.jpg'
        file_bytes_content = file.file.read()

        image_file_path = os.path.join(ROOT_DOWNLOAD_URL, file_name)
        if not os.path.exists(ROOT_DOWNLOAD_URL):
            os.mkdir(ROOT_DOWNLOAD_URL)

        with open(image_file_path, 'wb') as img_file_ptr:
            img_file_ptr.write(file_bytes_content)

        input_data = InputModel(
            image_file=image_file_path,
            back_url=None,
            threshold=threshold)
        task = InferenceProcessTask(inference_detector,
                                    input_data=input_data,
                                    cls_threshold=threshold)
        task.input_data_content_type = file.content_type
        task.run()
        response_data = task.response_data
        if display_image_only:
            return FileResponse(path=image_file_path,
                                media_type=file.content_type)
    except Exception as e:
        print(e)
        response_data["code"] = "failed"
        response_data["msg"] = "Failed to run inference on image"
    finally:
        # remove cached file
        if os.path.exists(image_file_path):
            os.remove(image_file_path)

    return response_data


@app.post("/inference_model_url")
async def img_object_detection(*, input_data: InputModel,
                               background_tasks: BackgroundTasks):
    response_data = dict()
    try:
        threshold = input_data.threshold
        task = InferenceProcessTask(
            inference_detector,
            input_data=input_data,
            threshold=threshold)

        if input_data.back_url is None:
            task.run()
            # if the user doesn't provide returning url, we just wait
        else:
            background_tasks.add_task(task.run)
        response_data = task.response_data

    except Exception as e:
        print(e)
        response_data["code"] = "failed"

    return response_data


async def cache_image(name: str,
                      data: bytes):
    # hardcoded values
    print("Cached the Image")
    root_dir = os.path.abspath(os.path.curdir)
    cache_dir = os.path.join(root_dir, ".dataCache")
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    os.chdir(cache_dir)
    with open(name, 'bw') as f:
        f.write(data)
    os.chdir(root_dir)
    return os.path.join(cache_dir, name)


@app.get("/inference_model")
def index_root():
    print("Acessing the Root Path")
    return {"Usage": "pick the model name you want to use, and upload the image",
            "Returns": "Image processed by model"}


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
