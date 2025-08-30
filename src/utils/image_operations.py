import cv2
import numpy as np


def pad_resize_image(cv2_img: np.ndarray,
                     new_size: tuple[int, int] = (640, 480),
                     color: tuple[int, int, int] = (125, 125, 125)) -> np.ndarray:
    """Resize and pad image with color if necessary, maintaining orig scale.

    Args:
        cv2_img: numpy.ndarray = cv2 image
        new_size: tuple(int, int) = (width, height)
        color: tuple(int, int, int) = (B, G, R)

    """
    in_h, in_w = cv2_img.shape[:2]
    new_w, new_h = new_size
    # rescale down
    scale = min(new_w / in_w, new_h / in_h)
    # get new sacled widths and heights
    scale_new_w, scale_new_h = int(in_w * scale), int(in_h * scale)
    resized_img = cv2.resize(cv2_img, (scale_new_w, scale_new_h))
    # calculate deltas for padding
    d_w = max(new_w - scale_new_w, 0)
    d_h = max(new_h - scale_new_h, 0)
    # center image with padding on top/bottom or left/right
    top, bottom = d_h // 2, d_h - (d_h // 2)
    left, right = d_w // 2, d_w - (d_w // 2)
    pad_resized_img = cv2.copyMakeBorder(resized_img,
                                         top, bottom, left, right,
                                         cv2.BORDER_CONSTANT,
                                         value=color)
    return pad_resized_img


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, np.ndarray):
        boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
        boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
        boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
        boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2
    else:  # torch.Tensor
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


if __name__ == "__main__":
    # sample usage
    h, w = (360, 480)    # orig input image size
    mw, mh = (224, 224)  # model input size with necessary padding

    boxes = np.array([[50, 50, 100, 100], [80, 80, 120, 120]])
    # rescale detections to orig image size taking the padding into account
    boxes = boxes * np.array([mw, mh, mw, mh])
    boxes = scale_coords((mh, mw), boxes, (h, w)).round()
    for i, box in enumerate(boxes):
        # correct xyxy bbox coords for the orig image
        xmin, ymin, xmax, ymax = box.astype('int')
