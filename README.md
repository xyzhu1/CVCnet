# CVCnet
*Xiangyuan Zhu; Kehua Guo; Sheng Ren; Bin Hu; Min Hu; Hui Fang. "Cross View Capture for Stereo Image
Super-Resolution"*,  in IEEE Transactions on Multimedia [IEEExplore](https://ieeexplore.ieee.org/document/9465749)

## Requirements
- Python 3.6 (Anaconda is recommended)
- skimage
- imageio
- Pytorch 1.4.0
- torchvision  0.5.0
- tqdm 
- pandas
- cv2 (pip install opencv-python)

## Test
### Prepare test data
1. Download the [KITTI2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) dataset and put folders `testing/colored_0` and `testing/colored_1` in `data/test/KITTI2012/original` 
2. Cd to `data/test` and run `generate_testset.m` to generate test data.
3. (optional) You can also download KITTI2015, Middlebury or other stereo datasets and prepare test data in `data/test` as below:
```
  datasets
  └── test
      ├── dataset_1
            ├── hr
                ├── scene_1
                      ├── hr0.png
                      └── hr1.png
                ├── ...
                └── scene_M
            └── lr_x4
                ├── scene_1
                      ├── lr0.png
                      └── lr1.png
                ├── ...
                └── scene_M
      ├── ...
      └── dataset_N
```

### Demo
```bash
python cvcnet_test.py 
```

## Train
```bash
python cvcnet_train.py 
``` 
