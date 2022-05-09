# deblur
This is an unofficial Python implementation of the 2014 IEEE paper "Deblurring Text Images via L0 Regularized Intensity and Gradient Prior" by [J. Pan, Z. Hu, Z. Su, and M. Yang](https://openaccess.thecvf.com/content_cvpr_2014/papers/Pan_Deblurring_Text_Images_2014_CVPR_paper.pdf)


## Project info
This is implemented as the final project for CS1430 at Brown. 
Contributing members:
Alan Gu
Edward Xing
Luca Fonstad
Paul Zhiyuan Zhou


## Setting up
```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```


## Running
```python
python3 motion_deblur.py  # deblur an image
python3 eval.py  # evaluate deblur results
```


## File structure
`data/`: the data directory
`conjugate_gradient.py`: our implementation of the conjugate gradient algorithm as an optimization scheme
`eval.py`: the script to evaluate results
`get_data.py`: data preprocessing and parsing
`motion_deblur.py`: the main magic and algorithms to deblur images
`params.py`: storage for hyperparameters