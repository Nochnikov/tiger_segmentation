import torch
import matplotlib.pyplot as plt
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
import numpy as np
from PIL import Image
#сюда изображение закидываешь

def receive_segmented_data(photo_origin):
    buf = io.BytesIO()
    with Image.open(photo_origin) as img:
        img.save(buf, 'JPEG')
    image_data = buf.getvalue()
    buf.close()

    img_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    test = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize_transform = A.Compose([A.Normalize(mean, std), ToTensorV2()])

    train_transform = A.Compose([
        A.ShiftScaleRotate(0.2, 0.2, 45),
        A.RGBShift(25, 25, 25),
        A.RandomBrightnessContrast(0.3, 0.3),
        normalize_transform
    ])
    test_transform = normalize_transform

    img_size = 128

    model = smp.Unet(encoder_name="efficientnet-b5", encoder_weights="imagenet", in_channels=3, classes=1, )
    state = torch.load(r'AI/best_model.pt', map_location=torch.device('cpu'))
    model.load_state_dict(state)
    device = 'cpu'
    test_source = cv2.resize(test, (img_size, img_size))
    test_source = cv2.cvtColor(test_source, cv2.COLOR_BGR2RGB)
    test_image = test_transform(image=test_source)['image']
    test_image = test_image.unsqueeze(0)

    test_image = test_image.to(device)

    y_pred = torch.sigmoid(model.forward(test_image))
    threshold = 0.5
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    fig, ax = plt.subplots(1, 2, dpi=200)
    ax[0].imshow(test_source)
    ax[0].imshow(y_pred[0, 0, :, :].detach().cpu().numpy(), alpha=0.4)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    image_data = buf.getvalue()
    buf.close()


    img_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    x, y, w, h = 160, 275, img.shape[1] - 850, img.shape[0] - 550
    cropped_img = img[y:y + h, x:x + w]


    buf = io.BytesIO()
    _, img_encoded = cv2.imencode('.png', cropped_img)
    buf.write(img_encoded)
    image_data = buf.getvalue()
    buf.close()


    return image_data




