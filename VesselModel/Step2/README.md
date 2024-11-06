This folder includes model implementation.



```
pip install transformers
```



Approach 1:
  for each epoch, train transformer and gru model separately.
  using data "data/Features/feature3.csv"

Steps:

1. configuration is saved in config.py.
2. GRU-MLP and approach 2 model implementation is in model.py
3. run python3 train_v1.py, checkpoint will save in data/Checkpoints/
4. model_valid.py: select random trip in test dataset and plot the actual and predicted value.
