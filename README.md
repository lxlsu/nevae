
#

## Requirements
-`tensorflow 1.4.1`

-`rdkit >= 2016.03.4`

-`networkx 2.0`

## Train model 

  `mkdir output_with_mask`

  `mkdir sample_with_mask`

  `mkdir zspace_with_mask`

  `echo "0" > output_with_mask/iteration.txt`

  `./batching.sh 100`



## Sample data   

  `python sample.py --num_epochs=10 --learning_rate=0.005 --log_every 100 --graph_file=data/ --out_dir=output_with_mask/ --sample_file sample_with_mask/ --z_dir zspace_with_mask/ --random_walk 5 --z_dim 7 --nodes 30 --edges 30 --mask_weight True >output_with_mask/nohup_sample.out`

## sample from already trained sample model

  `python sample.py --num_epochs=10 --learning_rate=0.005 --log_every 100 --graph_file=data/ --out_dir=model/ZINC/output_with_mask/ --sample_file sample_with_mask/ --z_dir zspace_with_mask/ --random_walk 5 --z_dim 7 --nodes 30 --edges 30 --mask_weight True > nohup_sample.out`

## checkvalidity of the molecules
  `mkdir figure_with_out_mask
  python checkvalidity.py sample_with_mask/ test.mol2 30 1 figure_with_mask/`

## check novelty
  `python novelty.py dataset.smi smiles.smi`


#### 20191219
1. 经过input_layer， 向量为k * nodes * d, 5 * nodes(30) * d(4)，
2. 经过unstak, 已经concat之后， 全连接层输入shape为 nodes*(k * d), 30 * 20
3 . ```list_cx = tf.unstack(c_x)

                print("list_cx.shape", list_cx)  # 5*(30*4)
                # output will be of shape n X kd
                # concat之后输入向量变成了30*20
                enc_hidden = fc_layer(tf.concat(list_cx, 1), k * d, activation=tf.nn.relu, scope="hidden")
    ```
