# Online experiments for RotatE 

Code for RotatE experiments from the EMNLP-Findings paper -- [Probabilistic Case-based Reasoning for Open-World Knowledge Graph Completion
](https://arxiv.org/abs/2010.03548)

```bash
# WN18RR

PCENT=0.1
python -u codes/run.py --cuda \
    --do_train \
    --do_valid \
    --do_test \
    --data_path data/wn18rr/ \
    --model RotatE \
    -n 1024 -b 512 -d 500 \
    -g 6.0 -a 0.5 -adv \
    -lr 0.00005 --max_steps 50000 \
    --ft_steps 10000 --valid_steps 5000 \
    -save saved_models/RotatE_wn18rr_nbh_"$PCENT"_smart_init --test_batch_size 8 \
    -de \
    --ft_all_entity_embeddings \
    --smart_init \
    --stream_init_proportion 0.5 \
    --n_stream_updates 10 \
    --frac_old_train_samples $PCENT \
    --sample_nbh \
    --stream_seed 42
```

```bash
# FB122

PCENT=0.1
python -u codes/run.py --cuda \
    --do_train \
    --do_valid \
    --do_test \
    --data_path data/FB122/ \
    --test_file_name testI.txt \
    --model RotatE \
    -n 256 -b 1024 -d 1000 \
    -g 9.0 -a 1.0 -adv \
    -lr 0.00005 --max_steps 60000 \
    --ft_steps 5000 --valid_steps 5000 \
    -save saved_models/RotatE_fb122_nbh_"$PCENT"_smart_init --test_batch_size 16 \
    -de \
    --ft_all_entity_embeddings \
    --smart_init \
    --stream_init_proportion 0.5 \
    --n_stream_updates 10  \
    --frac_old_train_samples $PCENT \
    --sample_nbh \
    --stream_seed 42
```

To freeze old embeddings: skip the flag `--ft_all_entity_embeddings`

To use random initialization of new entity embeddings: skip the flag `--smart_init`

To use a different percentage of the old training triples: set `PCENT` to the appropriate fraction

# RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
**Introduction**

This is the PyTorch implementation of the [RotatE](https://openreview.net/forum?id=HkgEQnRqYQ) model for knowledge graph embedding (KGE). We provide a toolkit that gives state-of-the-art performance of several popular KGE models. The toolkit is quite efficient, which is able to train a large KGE model within a few hours on a single GPU.

A faster multi-GPU implementation of RotatE and other KGE models is available in [GraphVite](https://github.com/DeepGraphLearning/graphvite).

**Implemented features**

Models:
 - [x] RotatE
 - [x] pRotatE
 - [x] TransE
 - [x] ComplEx
 - [x] DistMult

Evaluation Metrics:

 - [x] MRR, MR, HITS@1, HITS@3, HITS@10 (filtered)
 - [x] AUC-PR (for Countries data sets)

Loss Function:

 - [x] Uniform Negative Sampling
 - [x] Self-Adversarial Negative Sampling

**Usage**

Knowledge Graph Data:
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

**Train**

For example, this command train a RotatE model on FB15k dataset with GPU 0.
```
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/FB15k \
 --model RotatE \
 -n 256 -b 1024 -d 1000 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.0001 --max_steps 150000 \
 -save models/RotatE_FB15k_0 --test_batch_size 16 -de
```
   Check argparse configuration at codes/run.py for more arguments and more details.

**Test**

    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

**Reproducing the best results**

To reprocude the results in the ICLR 2019 paper [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://openreview.net/forum?id=HkgEQnRqYQ), you can run the bash commands in best_config.sh to get the best performance of RotatE, TransE, and ComplEx on five widely used datasets (FB15k, FB15k-237, wn18, wn18rr, Countries).

The run.sh script provides an easy way to search hyper-parameters:

    bash run.sh train RotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 200000 16 -de

**Speed**

The KGE models usually take about half an hour to run 10000 steps on a single GeForce GTX 1080 Ti GPU with default configuration. And these models need different max_steps to converge on different data sets:

| Dataset | FB15k | FB15k-237 | wn18 | wn18rr | Countries S* |
|-------------|-------------|-------------|-------------|-------------|-------------|
|MAX_STEPS| 150000 | 100000 | 80000 | 80000 | 40000 | 
|TIME| 9 h | 6 h | 4 h | 4 h | 2 h | 

**Results of the RotatE model**

| Dataset | FB15k | FB15k-237 | wn18 | wn18rr |
|-------------|-------------|-------------|-------------|-------------|
| MRR | .797 ± .001 | .337 ± .001 | .949 ± .000 |.477 ± .001
| MR | 40 | 177 | 309 | 3340 |
| HITS@1 | .746 | .241 | .944 | .428 |
| HITS@3 | .830 | .375 | .952 | .492 |
| HITS@10 | .884 | .533 | .959 | .571 |

**Using the library**

The python libarary is organized around 3 objects:

 - TrainDataset (dataloader.py): prepare data stream for training
 - TestDataSet (dataloader.py): prepare data stream for evluation
 - KGEModel (model.py): calculate triple score and provide train/test API

The run.py file contains the main function, which parses arguments, reads data, initilize the model and provides the training loop.

Add your own model to model.py like:
```
def TransE(self, head, relation, tail, mode):
    if mode == 'head-batch':
        score = head + (relation - tail)
    else:
        score = (head + relation) - tail

    score = self.gamma.item() - torch.norm(score, p=1, dim=2)
    return score
```

**Citation**

If you use the codes, please cite the following [paper](https://openreview.net/forum?id=HkgEQnRqYQ):

```
@inproceedings{
 sun2018rotate,
 title={RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space},
 author={Zhiqing Sun and Zhi-Hong Deng and Jian-Yun Nie and Jian Tang},
 booktitle={International Conference on Learning Representations},
 year={2019},
 url={https://openreview.net/forum?id=HkgEQnRqYQ},
}
```
