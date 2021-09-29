# DeepHop

Supporting Information for the paper "[Deep Scaffold Hopping with Multimodal Transformer Neural Networks](https://www.nature.com/articles/s42256-020-0152-y)"

DeepHop is a multi-modal molecular transformation framework. It accepts a hit molecule and an interest target protein sequence as inputs and design isofunctional molecular structures to the source compound.

![DeepHop](Image/DeepHop.png)



## Installation

Create a conda environment for QSAR-scorer:
```shell script
conda create env -f=score/env.yaml
```

Create a conda environment for Deephop:
```shell script
conda create env -f=deephop/env.yaml
```
Note that you should replace three source files(batch.py, field.py, iterator.py) of the torchtext library in "your deep hop env path/python3.7/site-packages/torchtext/data"  with the corrsponding three files contained in "deephop/replace_torchtext" since we have modified the codes.


## Scaffold hopping pairs construction
For the convenience of illustration, We assume that:
you code extract in  /data/u1/projects/mget_3d
environment for deephop is named deephop_env

```shell script
cd /data/u1/projects/mget_3d
conda activate deephop_env
```
you can use make_pair.py to generate hopping pairs.

## Dataset split
```shell script
python split_data.py -out_dir data40_tue_3d/0.60 -protein_group  data40 -target_uniq_rate 0.6 -hopping_pairs_dir hopping_pairs_with_scaffold
```

## Data preprocessing
```shell script
python preprocess.py -train_src data40_tue_3d/0.60/src-train.txt -train_tgt data40_tue_3d/0.60/tgt-train.txt -train_cond data40_tue_3d/0.60/cond-train.txt -valid_src data40_tue_3d/0.60/src-val.txt -valid_tgt data40_tue_3d/0.60/tgt-val.txt -valid_cond data40_tue_3d/0.60/cond-val.txt -save_data data40_tue_3d/0.60/seqdata -share_vocab -src_seq_length 1000 -tgt_seq_length 1000 -src_vocab_size 1000 -tgt_vocab_size 1000 -with_3d_confomer
```

## Model training
```shell script
python train.py -condition_dim 768  -use_graph_embedding -arch after_encoding -data data40_tue_3d/0.60/seqdata -save_model experiments/data40_tue_3d/after/models/model -seed 42 -save_checkpoint_steps 158 -keep_checkpoint 400 -train_steps 95193 -param_init 0 -param_init_glorot -max_generator_batches 32 -batch_size 8192 -batch_type tokens -normalization tokens -max_grad_norm 0 -accum_count 4 -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 475 -learning_rate 2 -label_smoothing 0.0 -report_every 10 -layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer -dropout 0.1 -position_encoding -share_embeddings -global_attention general -global_attention_function softmax -self_attn_type scaled-dot -heads 8 -transformer_ff 2048 -log_file experiments/data40_tue_3d/after/train.log -tensorboard -tensorboard_log_dir experiments/data40_tue_3d/after/logs -world_size 4 -gpu_ranks 0 1 2 3 -valid_steps 475 -valid_batch_size 32
```

## Hops generation
To generate the output SMILES by loading saved model
```shell script
python translate.py -condition_dim 768  -use_graph_embedding -arch after_encoding -with_3d_confomer -model /data/u1/projects/mget_3d/experiments/data40_tue/3d_gcn/models/model_step_9500.pt -gpu 0 -src data40_tue_3d/src-test.txt -cond data40_tue_3d/cond-test.txt -output /data/u1/projects/mget_3d/summary_tue/data40/after/9500/pred.txt -beam_size 10 -n_best 10 -batch_size 16 -replace_unk -max_length 200 -fast -use_protein40
```

## Evaluation

To evaluate our model
```shell script
 python pvalue_score_predictions.py -beam_size 10 -src summary_tue/data40/after/9500/src-test-protein.txt -prediction /data/u1/projects/mget_3d/summary_tue/data40/after/9500/pred.txt -score_file /data/u1/projects/mget_3d/summary_tue/data40/after/9500/score.csv -invalid_smiles -cond summary_tue/data40/after/9500/cond-test-protein.txt -train_data_dir /data/u1/projects/mget_3d/data40_tue_3d -scorer_model_dir /data/u1/projects/score/total_mtr -pvalue_dir /data/u1/projects/mget_3d/score_train_data
```
where the final result report is saved at /data/u1/projects/mget_3d/summary_tue/data40/after/9500/score_final.csv




