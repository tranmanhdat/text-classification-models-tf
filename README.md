# Text Classification Models with Tensorflow
Tensorflow implementation of Text Classification Models.

Implemented Models:
Very Deep CNN [[paper](https://arxiv.org/abs/1606.01781)]

## Requirements
- Python3
- Tensorflow
- pip install -r requirements.txt

## Usage

### 1. run 1_prepare_text.py to make text file : python 1_prepare_text.py ../data_split/ ../data_merger/
### 2. run 2_get_alphabet.py to get alphabet :
python 2_get_alphabet.py ../data_merger/train.txt ../data_merger/alphabet1.txt
python 2_get_alphabet.py ../data_merger/test.txt ../data_merger/alphabet2.txt

### Train
To train classification models for dbpedia dataset,
```
$ python train.py --model="<MODEL>"
```
(\<Model>: word_cnn | char_cnn | vd_cnn | word_rnn | att_rnn | rcnn)

### Test
To test classification accuracy for test data after training,
```
$ python test.py --model="<TRAINED_MODEL>"
```

## Models


### Char-level CNN
Implementation of [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626).

<img width="600" src="https://user-images.githubusercontent.com/6512394/41590359-c6c94f8a-73f1-11e8-8bda-976ddf09e817.png">



## References
- [dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)
- [zonetrooper32/VDCNN](https://github.com/zonetrooper32/VDCNN)
