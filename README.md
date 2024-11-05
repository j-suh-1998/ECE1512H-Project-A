# ECE1512H-Project-A
Jangwon Suh

Electrical and Computer Engineering, University of Toronto

Refer to the ECE1512H_ProjectA_JangwonSuh_v1.pdf file


1. **Task 1-2-(a)**
For MNIST dataset: edit ./DataDAM/task_1_2_a_MNIST.py file's
```bash
train_dataset = datasets.MNIST(root='./mnist_dataset', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_dataset', train=False, transform=transform, download=True)
```
to anywhere you want, then
```bash
cd DataDAM
python task_1_2_a_MNIST.py
```

For MHIST dataset: edit ./DataDAM/task_1_2_a_MHIST.py file's
```bash
root_dir = './mhist_dataset/'
csv_file = './mhist_dataset/annotations.csv'
```
to anywhere you want, then
```bash
cd DataDAM
python task_1_2_a_MHIST.py
```


2. **Task 1-2-(b)~(c)**
For MNIST dataset: edit ./DataDAM/utils.py file's line 320 to
```bash
net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
```
then
```bash
cd DataDAM
python main_DataDAM.py --dataset=MNIST --ipc=10 --eval_mode=SS --num_eval=100 --batch_real=256 --batch_train=256 --init=real --data_path=wherever_you_want --save_path=wherever_you_want
```

For MHIST dataset: edit ./DataDAM/utils.py file's line 320 to
```bash
net_width, net_depth, net_act, net_norm, net_pooling = 32, 7, 'relu', 'instancenorm', 'avgpooling'
```
then
```bash
cd DataDAM
python main_DataDAM.py --dataset=MHIST --ipc=50 --eval_mode=SS --num_eval=200 --batch_real=128 --batch_train=128 --init=real --data_path=wherever_you_want --save_path=wherever_you_want
```


3. **Task 1-2-(d)**
For MNIST dataset: edit ./DataDAM/utils.py file's line 320 to
```bash
net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
```
then
```bash
cd DataDAM
python main_DataDAM.py --dataset=MNIST --ipc=10 --eval_mode=SS --num_eval=100 --batch_real=256 --batch_train=256 --init=noise --data_path=wherever_you_want --save_path=wherever_you_want
```

For MHIST dataset: edit ./DataDAM/utils.py file's line 320 to
```bash
net_width, net_depth, net_act, net_norm, net_pooling = 32, 7, 'relu', 'instancenorm', 'avgpooling'
```
then
```bash
cd DataDAM
python main_DataDAM.py --dataset=MHIST --ipc=50 --eval_mode=SS --num_eval=200 --batch_real=128 --batch_train=128 --init=noise --data_path=wherever_you_want --save_path=wherever_you_want
```


4. **Task 1-3**
For MNIST dataset: edit ./DataDAM/utils.py file's line 320 to
```bash
net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
```
then
```bash
cd DataDAM
python main_DataDAM.py --dataset=MNIST --ipc=10 --eval_mode=M --num_eval=100 --batch_real=256 --batch_train=256 --init=real --data_path=wherever_you_want --save_path=wherever_you_want
```

For MHIST dataset: edit ./DataDAM/utils.py file's line 320 to
```bash
net_width, net_depth, net_act, net_norm, net_pooling = 32, 7, 'relu', 'instancenorm', 'avgpooling'
```
then
```bash
cd DataDAM
python main_DataDAM.py --dataset=MHIST --ipc=50 --eval_mode=M --num_eval=200 --batch_real=128 --batch_train=128 --init=real --data_path=wherever_you_want --save_path=wherever_you_want
```


5. **Task 1-4**
Put your generated synthetic MNIST dataset from task 1-2-(b)~(c) in ./dc_benchmark/distilled_results/DC/synthetic/IPC10 and rename it as res_DC_synthetic_ConvNet_10ipc.pt
then
```bash
cd dc_benchmark/evaluator
python evaluator.py
```


6. **Task 2-2**
For real image initialization:
```bash
cd PAD/buffer
python buffer_CL.py
cd ../distill
python PAD_depth.py --cfg ../configs/MNIST/ConvIN/IPC10.yaml
```

For Gaussian noise initialization:
```bash
cd PAD/buffer
python buffer_CL.py
cd ../distill
python PAD_depth.py --cfg ../configs/MNIST/ConvIN/IPC10_noise.yaml
```


7. **Optional**
For IPC = 1:
```bash
cd DATM/buffer
python buffer_FTD.py
cd ../distill
python DATM_testla.py --cfg ../configs/MNIST/ConvIN/IPC1.yaml
```

For IPC = 10:
```bash
cd DATM/buffer
python buffer_FTD.py
cd ../distill
python DATM_testla.py --cfg ../configs/MNIST/ConvIN/IPC1.yaml
```

For IPC = 100:
```bash
cd DATM/buffer
python buffer_FTD.py
cd ../distill
python DATM_testla.py --cfg ../configs/MNIST/ConvIN/IPC1.yaml
```


## Acknowledgement
This code is built upon [DataDAM](https://github.com/DataDistillation/DataDAM.git), [DATM](https://github.com/NUS-HPC-AI-Lab/DATM.git), [dc_benchmark](https://github.com/justincui03/dc_benchmark.git), [PAD](https://github.com/NUS-HPC-AI-Lab/PAD.git)
