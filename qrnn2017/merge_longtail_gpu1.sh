#!/bin/bash

SCRIPT_ROOT=./dev/ltr
DATA_ROOT=../webrank_relevance/data_uniq_longtail
GPU_ID=1
EXTRA_WDICT=../webrank_relevance/word_list/vocab_new_hidden_775354.pkl
RESULT_DIR=$1
CKPT_PREFIX=$2
ITER_NUM=$3

#python $SCRIPT_ROOT/extract_input.py < $DATA_ROOT/new_data > $DATA_ROOT/fake_input_$ITER_NUM
CUDA_VISIBLE_DEVICES=$GPU_ID python srun_log.py --ckpt_path=$RESULT_DIR/"$CKPT_PREFIX"_"$ITER_NUM" --extra_wdict=$EXTRA_WDICT --extra_testfnms=$DATA_ROOT/fake_input.txt 2>$DATA_ROOT/fake_output_$ITER_NUM
python $SCRIPT_ROOT/checker.py $DATA_ROOT $DATA_ROOT/fake_output_$ITER_NUM > $DATA_ROOT/"$CKPT_PREFIX"_"$ITER_NUM".res.fea 2>$DATA_ROOT/"$CKPT_PREFIX"_"$ITER_NUM".count.log
python $SCRIPT_ROOT/featuremerge.py $DATA_ROOT/"$CKPT_PREFIX"_"$ITER_NUM".res.fea $DATA_ROOT/merge_data > $DATA_ROOT/"$ITER_NUM".ltr
python $SCRIPT_ROOT/RankbyFeatureComp.py $SCRIPT_ROOT/labeled.docid -f 6004 $SCRIPT_ROOT/featureName $DATA_ROOT/"$ITER_NUM".ltr 100 $RESULT_DIR/"$CKPT_PREFIX"_"$ITER_NUM".longtail.compare $DATA_ROOT/detail."$ITER_NUM".ltr unlabel lookup
rm -f $DATA_ROOT/fake_output_$ITER_NUM $DATA_ROOT/"$CKPT_PREFIX"_"$ITER_NUM".res.fea $DATA_ROOT/"$CKPT_PREFIX"_"$ITER_NUM".count.log $DATA_ROOT/"$ITER_NUM".ltr $DATA_ROOT/detail."$ITER_NUM".ltr
