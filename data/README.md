---
license: cc-by-4.0
tags:
- automatic-rigging
size_categories:
- 10K<n<100K
---
<div align="center">

<h1>MagicArticulate: Make Your 3D Models Articulation-Ready</h1>

<p>
  <a href="https://chaoyuesong.github.io"><strong>Chaoyue Song</strong></a><sup>1,2</sup>,
  <a href="http://jeff95.me/"><strong>Jianfeng Zhang</strong></a><sup>2*</sup>,
  <a href="https://lixiulive.com/"><strong>Xiu Li</strong></a><sup>2</sup>,
  <a href="https://scholar.google.com/citations?user=afDvaa8AAAAJ&hl"><strong>Fan Yang</strong></a><sup>1,2</sup>,
  <a href="https://buaacyw.github.io/"><strong>Yiwen Chen</strong></a><sup>1</sup>,
  <a href="https://zcxu-eric.github.io/"><strong>Zhongcong Xu</strong></a><sup>2</sup>,
 <br>
  <a href="https://liewjunhao.github.io/"><strong>Jun Hao Liew</strong></a><sup>2</sup>,
  <strong>Xiaoyang Guo</strong><sup>2</sup>,
  <a href="https://sites.google.com/site/fayaoliu"><strong>Fayao Liu</strong></a><sup>3</sup>,
  <a href="https://scholar.google.com.sg/citations?user=Q8iay0gAAAAJ"><strong>Jiashi Feng</strong></a><sup>2</sup>,
  <a href="https://guosheng.github.io/"><strong>Guosheng Lin</strong></a><sup>1*</sup>
  <br>
  *Corresponding authors
  <br>
    <sup>1 </sup>Nanyang Technological University
  <sup>2 </sup>Bytedance Seed
  <sup>3 </sup>A*STAR
</p>

<h3>CVPR 2025</h3>

<p>
  <a href="https://chaoyuesong.github.io/MagicArticulate/"><strong>Project</strong></a> |
  <a href="https://chaoyuesong.github.io/files/MagicArticulate_paper.pdf"><strong>Paper</strong></a> |
  <a href="https://github.com/Seed3D/MagicArticulate"><strong>Code</strong></a> |
  <a href="https://www.youtube.com/watch?v=eJP_VR4cVnk"><strong>Video</strong></a>
</p>


</div>

<br />

### Update
- 2025.8.22: We release the diverse-pose subset (10.8k for training, 500 for testing) of Articulation-XL2.0 introduced in our latest project [Puppeteer](https://chaoyuesong.github.io/Puppeteer/). You can combine this subset with the main set for training.
- 2025.4.18: We have updated the preprocessed dataset to exclude entries with skinning issues (118 from the training and 3 from the test, whose skinning weight row sums fell below 1) and duplicated joint names (2 from the training). You can download the [cleaned data](https://huggingface.co/datasets/chaoyue7/Articulation-XL2.0) again or update it yourself by running: `python data_utils/update_npz_rm_issue_data.py`. Still remember to normalize skinning weights in your dataloader.
- 2025.3.22: Update the preprocessed data to include vertex normals. The data size will increase by 10GB after adding the normals. To save data load time during training, we suggest removing some unused information.
- 2025.3.20: Release Articulation-XL2.0 preprocessed data (a NPZ file includes vertices, faces, joints, bones, skinning weights, uuid, etc.).
- 2025.2.16: Release metadata for Articulation-XL2.0!

### Overview
This repository introduces <b>Articulation-XL2.0</b>, a large-scale dataset featuring over <b>48K</b> 3D models with high-quality articulation annotations, filtered from Objaverse-XL. Compared to version 1.0, Articulation-XL2.0 includes 3D models with multiple components. For further details, please refer to the statistics below.
<p align="center">
  <img width="60%" src="https://raw.githubusercontent.com/Seed3D/MagicArticulate/main/assets/data_statistics.png"/>
</p>
Note: The data with rigging has been deduplicated (over 150K). The quality of most data has been manually verified.

<p align="center">
  <img width="80%" src="https://raw.githubusercontent.com/Seed3D/MagicArticulate/main/assets/articulation-xl2.0.png"/>
</p>

### Metadata
We provide the following information in the metadata of Articulation-XL2.0.
```
uuid,source,vertex_count,face_count,joint_count,bone_count,category_label,fileType,fileIdentifier
```

### Preprocessed data
We provide the preprocessed data that saved in NPZ files, which contain the following information:
```
'vertices', 'faces', 'normals', 'joints', 'bones', 'root_index', 'uuid', 'pc_w_norm', 'joint_names', 'skinning_weights_value', 'skinning_weights_row', 'skinning_weights_col', 'skinning_weights_shape'
```

### Citation

```
@inproceedings{song2025magicarticulate,
      title={MagicArticulate: Make Your 3D Models Articulation-Ready}, 
      author={Chaoyue Song and Jianfeng Zhang and Xiu Li and Fan Yang and Yiwen Chen and Zhongcong Xu and Jun Hao Liew and Xiaoyang Guo and Fayao Liu and Jiashi Feng and Guosheng Lin},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2025},
}

@article{song2025puppeteer,
  title={Puppeteer: Rig and Animate Your 3D Models},
  author={Chaoyue Song and Xiu Li and Fan Yang and Zhongcong Xu and Jiacheng Wei and Fayao Liu and Jiashi Feng and Guosheng Lin and Jianfeng Zhang},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```