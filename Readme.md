# An implementation for paper [Topic Segmentation for Dialogue Stream](https://ieeexplore.ieee.org/document/9023126)

Requirements:
* python3.7
* pytorch1.4
* [huggingface transformers](https://github.com/huggingface/transformers)
* download the pretrained bert-base-chinese pytorch model from huggingface, and put it under the ./model/bert-base-chinese

Run the training & testing process with 

```python
python3 main.py
```

The hyperparameters of the model can be changed in config.py.

The weibo dataset is copied from the original author's [link](https://github.com/zll17/Topic_Seg_BERT_TCN)