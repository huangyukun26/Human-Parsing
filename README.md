# Human Parsing

### Requirements

python 3.6  
PyTorch 0.4.1

using anaconda

```conda
conda env create -f environment.yaml
```

#### Tip for Requirements

Ubuntu20.04 system is recommended

If you are using a GPU of 3060 and above, please use Python 3.8 with the corresponding Pytorch and other dependencies.

If you are using cloud server,Possible workaround for the GPU part:

```python
#saved_state_dict = torch.load(args.restore_from)----->
saved_state_dict = torch.load(args.restore_from, map_location="cpu")

#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu----->
os.environ["CUDA_VISIBLE_DEVICES"] = "your gpu's number"
```

### Dataset and pretrained model

Dataset: LIP and ATR

pretrained model: resnet 101

### Training and Evaluation

using python

Start training first

```python
python train.py
```

This will then saves a model, which in the snapshots_256_256/demo/epoch_0.pth.

To evaluate results plaese run this code

```python
python evaluate.py
```

using script (ce2p)

```bash
./run.sh
```

To evaluate the results, please download 'LIP_epoch_149.pth' from [baidu drive](https://pan.baidu.com/s/1NoxI_JetjSVa7uqgVSKdPw) or [Google drive](https://drive.google.com/open?id=1rzLU-wK6rEorCNJfwrmIu5hY2wRMyKTK), and put into snapshots directory.

```
./run_evaluate.sh
```
