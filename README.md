# Slovo - Russian Sign Language Dataset

## Models
We provide some pre-trained models as the baseline for Russian sign language recognition.
We tested models with frames number from [16, 32, 48], and **the best for each are below**.
The first number in the model name is frames number and the second is frame interval.

| Model Name        | Model Size (MB) | Metric | ONNX                                                                                                            | TorchScript                                                                                                 |
|-------------------|-----------------|--------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| MViTv2-small-16-4 | 140.51          | 58.35  | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit16-4.onnx) | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/pt/mvit16-4.pt) |
| MViTv2-small-32-2 | 140.79          | 64.09  | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit32-2.onnx) | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/pt/mvit32-2.pt) |
| MViTv2-small-48-2 | 141.05          | 62.18  | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit48-2.onnx) | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/pt/mvit48-2.pt) |
| Swin-large-16-3   | 821.65          | 48.04  | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/swin/onnx/swin16-3.onnx) | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/swin/pt/swin16-3.pt) |
| Swin-large-32-2   | 821.74          | 54.84  | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/swin/onnx/swin32-2.onnx) | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/swin/pt/swin32-2.pt) |
| Swin-large-48-1   | 821.78          | 55.66  | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/swin/onnx/swin48-1.onnx) | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/swin/pt/swin48-1.pt) |
| ResNet-i3d-16-3   | 146.43          | 32.86  | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/resnet/onnx/resnet16-3.onnx) | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/resnet/pt/resnet16-3.pt) |
| ResNet-i3d-32-2   | 146.43          | 38.38  | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/resnet/onnx/resnet32-2.onnx) | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/resnet/pt/resnet32-2.pt) |
| ResNet-i3d-48-1   | 146.43          | 43.91  | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/resnet/onnx/resnet48-1.onnx) | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/resnet/pt/resnet48-1.pt) |

## SignFlow models

| Model Name | Desc                                                                                                                | ONNX                                                                                                    | Params |
|------------|---------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|--------|
| SignFlow-A | **63.3 Top-1** Acc on  [WLASL-2000](https://paperswithcode.com/sota/sign-language-recognition-on-wlasl-2000) (SOTA) | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/SignFlow-A.onnx) | 36M    |
| SignFlow-R | Pre-trained on **~50000** samples, has **267** classes, tested with GigaChat (as-is and context-based modes)        | [weights](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/SignFlow-R.onnx) | 37M    |
