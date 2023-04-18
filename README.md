- [TAP-GNN](#tap-gnn)
    - [Setup the Environment](#setup-the-environment)
    - [Data Preprocess](#data-preprocess)
    - [Run the Script](#run-the-script)
        - [Temporal Link Prediction (Transductive)](#temporal-link-prediction-transductive)
        - [Online Link Prediction](#online-link-prediction)
        - [Temporal Node Classification](#temporal-node-classification)
    - [Cite us](#cite-us)

# TAP-GNN

Code for [Temporal Aggregation and Propagation Graph Neural Networks For Dynamic Representation](https://arxiv.org/abs/2304.07503), and we have updated the latest results of baseline methods in the arxiv version than the IEEE published version.
> Authors: Tongya Zheng, Xinchaon Wang, Zunlei Feng, Jie Song, Yunzhi Hao, Mingli Song, Xingen Wang, Xinyu Wang, Chun Chen

We utilize the cool feature `group_apply_edges` of [dgl-0.4.3post2](./resources/dgl_cu102-0.4.3.post2-cp36-cp36m-manylinux1_x86_64.whl) to compute the aggregated neighbor messages of nodes along the chronological order, reducing the computation complexify of temporal graph convolution from $O(\sum_{v \in V} d_v^2) \ge O(E\cdot \bar{d})$ to $O(E)$.
The proposed TAP-GNN reuses the historical node embeddings to compute new node embeddings, eliminating nodes' cross-time dependencies on historical neighbors.

The latest dgl version has been [dgl-1.0x](https://docs.dgl.ai/) by now and removed the slow operation `group_apply_edges` since 0.4.3.
We are trying to work out our TAP-GNN with the newly implemented [segment_reduce](https://docs.dgl.ai/generated/dgl.ops.segment_reduce.html#dgl.ops.segment_reduce) feature.

## Setup the Environment

- `conda create -n tap python=3.7 -y`

- `pip install -r requirements.txt`

- My torch version is `torch-1.9.1+cu102`

## Data Preprocess

We have preprocessed most temporal graphs in the `data/format_data` directory, and placed the JODIE datasets at [Google drive](https://drive.google.com/drive/folders/19ItQ4G64rYa6so1IQ6NxEq_Ok7K9Sqsp?usp=sharing), which can be downloaded and placed at the `data/format_data`.

```bash
bash init.sh
```
We use `init.sh` to make necessary directories for our experiments to store generated datasets by `data/*`, boost the training speed by `gumbel_cache` and `sample_cache`, record training details by `log`, record testing results by `results` and `nc-results`, save our trained models by `ckpt` and `saved_models`.

```bash
python data_unify.py -t datasplit
python data_unify.py -t datalabel
```
We use `-t datasplit` to split datasets into the training, validation and testing set according to the ratios.

## Run the Script

Running `bash ./init.sh` to create necessary directories.

### Temporal Link Prediction (Transductive)

- We have computed the target edge ids of the proposed Aggregation and Propagation (AP), where we store the temporal node embeddings with the edge-size hidden embeddings. It is the same with our analysis in the paper that the size of temporal node embeddings is at the same-level of temporal edges $O(E)$. In this way, the Aggregation operation acts as a message-passing operation, and the Propagation operation acts as a reduction function with the proper normalization term, which can be seen in [](file:///layers.py#)
> `python fast_gtc.py -d fb-forum`

### Online Link Prediction

- We have built an online model with the same functionality to the full-graph training model.
> `python online_gtc.py -d JODIE-wikipedia -bs 500 --n-layers 30`

### Temporal Node Classification

- We find that using trainable node embeddings for inductive learning harms the generalization ability. So we freeze the node embeddings as all zero vectors for JODIE datasets.
> `python fast_gtc.py -d JODIE-wikipedia --task node --no-trainable`
- We have set the balanced training strategy for the extremely imbalanced task of temporal node classification.
> `python node_model.py -d JODIE-wikipedia`


## Cite us
```
@article{zheng2023temporal,
  title={Temporal Aggregation and Propagation Graph Neural Networks For Dynamic Representation},
  author={Zheng, Tongya and Wang, Xinchao and Feng, Zunlei and Song, Jie and Hao, Yunzhi and Song, Mingli and Wang, Xingen and Wang, Xinyu and Chen, Chun},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2023},
  pages={1--15},
  publisher={IEEE}
}
```
