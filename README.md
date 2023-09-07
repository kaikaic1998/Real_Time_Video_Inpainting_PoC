<h1 align="center">Real Time Video Inpainting (PoC)</h1>

<p align="center">
Framework
<img src="./figures/framework.jpg" width=100% class="center">
</p>

<p align="center">
Demo
<table class="center">
  <tr>
    <td style="text-align:center">Input Image</td>
    <td style="text-align:center">Real Time Inpainting</td>
  </tr>
  <tr>
    <td><img src="./gif/car-turn.gif" width="100%"></td>
    <td><img src="./gif/car-turn_inpaint_result.gif" width="100%"></td>
  </tr>
  <tr>
    <td><img src="./gif/motorbike.gif" width="100%"></td>
    <td><img src="./gif/motorbike_inpaint_result.gif" width="100%"></td>
  </tr>
</table>
</p>

<h2 align="center">Installation</h2>
<p align="left">

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.
</p>

<h3 align="center">Clone the repository locally</h3>
<p align="left">

```
git clone https://github.com/kaikaic1998/Real_Time_Video_Inpainting_PoC.git
cd Real_Time_Video_Inpainting_PoC
```
```
pip install -r requirements.txt
cd Deep_Video_Inpainting/inpainting
bash install.sh
cd ../../
```
</p>

<h3 align="center">Pre-trained Model Checkoints</h3>
<p align="left">
Download the three pre-trained models
</p>

(**Light HQ-SAM** for real-time need): [ViT-Tiny HQ-SAM model.](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth)

(**Siammask**): [Siammask Model.](https://drive.google.com/file/d/1VDoG616hJ3GVywljX4pfdzLEMuCDqml-/view?usp=sharing)

(**Deep Video Inpainting**): [Inpainting Model.](https://drive.google.com/file/d/1Gr72-DYtY2vO6tIA9hw1Mj5-WpInmoFZ/view?usp=sharing)

<p align="left">
Put them in 'Model_CP' folder
</p>

<h3 align="center">Demo</h3>
<p align="left">

```
python run.py
```
</p>

<h2 align="center">Citation</h2>
<p align="left">

```
@article{sam_hq,
    title={Segment Anything in High Quality},
    author={Ke, Lei and Ye, Mingqiao and Danelljan, Martin and Liu, Yifan and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    journal = {arXiv:2306.01567},
    year = {2023}
} 
```
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
```
@ARTICLE{kim2020vipami,
  author={Kim, Dahun and Woo, Sanghyun and Lee, Joon-Young and Kweon, In So},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Recurrent Temporal Aggregation Framework for Deep Video Inpainting},
  year={2020},
  volume={42},
  number={5},
  pages={1038-1052},}
```
```
@inproceedings{wang2019fast,
    title={Fast online object tracking and segmentation: A unifying approach},
    author={Wang, Qiang and Zhang, Li and Bertinetto, Luca and Hu, Weiming and Torr, Philip HS},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    year={2019}
}
```
</p>