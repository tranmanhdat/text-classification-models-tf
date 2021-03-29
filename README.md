# Text Classification Models with Tensorflow
Tensorflow implementation of Text Classification Models.

Implemented Models:
Very Deep CNN [[paper](https://arxiv.org/abs/1606.01781)]
Implementation of [Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781).

<img height="600" src="https://user-images.githubusercontent.com/6512394/41590802-e68f71cc-73f2-11e8-88c6-4bf84bf3410e.png">

## Requirements
- Python3
- Tensorflow
- pip install -r requirements.txt

## Usage
#### Merger data from repo [craw_and_text_classification](https://github.com/tranmanhdat/craw_and_text_classification)
 run 1_prepare_text.py to make text file
```
python 1_prepare_text.py ../data_5/ ../data_merger/
```
you need specify alphabet by yourself, or use my define alphabet for
 Vietnamese in [data_utils.py](https://github.com/tranmanhdat/text-classification-models-tf/blob/5dbb15c393e338854fe08b28106e2cb581cb2f0e/data_utils.py#L33)
```
python 2_get_alphabet.py ../data_merger/train.txt ../data_merger/alphabet1.txt
python 2_get_alphabet.py ../data_merger/test.txt ../data_merger/alphabet2.txt
```
and merger by using 3_merger_alphabet.py or handmade  
To use my custom data, you can get it from [data_merger](https://drive.google.com/file/d/1sPz-7Rn7iViJ9mvQOZT0F9iKLkVcab61/view?usp=sharing)
#### Train
```
python train.py ../data_merger/train.txt
```
#### Test
To test classification accuracy for test data after training,
```
python test.py ../data_merger/test.txt
```
## References
- [dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)
- [zonetrooper32/VDCNN](https://github.com/zonetrooper32/VDCNN)
