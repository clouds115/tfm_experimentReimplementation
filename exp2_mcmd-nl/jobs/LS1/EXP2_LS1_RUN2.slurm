#!/bin/bash
#SBATCH --job-name=EXP2_LS1_RUN2
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=34:00:00
#SBATCH --output=/home/x241/x241844/claudio/projects/dnn_reimplementation_experiment/logs/exp2_LS1_run2.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=claudio.v.catalano@icloud.com

set -e
set -o pipefail

########################################
# PATHS
########################################

BASE_EXP=/home/x241/x241844/claudio/projects/dnn_reimplementation_experiment/exp2_mcmd-nl/LS1/run2
CCT5_DIR=/home/x241/x241844/claudio/projects/baselines_evaluation/CCT5
COME_DIR=/home/x241/x241844/claudio/projects/baselines_evaluation/come
EVAL_SCRIPT=/home/x241/x241844/claudio/projects/dnn_reimplementation_experiment/evaluate_simple.py

LANGUAGES=("PHP" "R" "Swift" "TypeScript" "Objective-C")

export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false

mkdir -p $BASE_EXP

########################################
# LOOP
########################################

for LANG in "${LANGUAGES[@]}"
do
    echo "======================================"
    echo "Processing $LANG"
    echo "======================================"

    ########################################
    # ========== CCT5 =====================
    ########################################

    source /media/apps/avx512-2021/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
    conda activate cct5
    cd $CCT5_DIR

    OUT_DIR=$BASE_EXP/CCT5/$LANG
    mkdir -p $OUT_DIR/predictions

    echo "Training CCT5 (MCMD-NL) - $LANG"

    bash scripts/finetune_msggen.sh \
        -g 0 \
        -l $LANG \
        -d /home/x241/x241844/claudio/projects/baselines_evaluation/CCT5/mcmd-nl_cct5

    ########################################
    # INFERENCE MCMD-NL
    ########################################

    echo "CCT5 Inference MCMD-NL - $LANG"

    bash scripts/finetune_msggen.sh \
        -g 0 \
        -l $LANG \
        -d /home/x241/x241844/claudio/projects/baselines_evaluation/CCT5/mcmd-nl_cct5 \
        -e outputs/models/fine-tuning/CommitMsgGeneration/$LANG/checkpoint-final/pytorch_model.bin

    CCT5_OUTPUT=$CCT5_DIR/outputs/models/fine-tuning/CommitMsgGeneration/$LANG/checkpoint-final

    # Copy the checkpoint to the output directory for record-keeping
    cp outputs/models/fine-tuning/CommitMsgGeneration/$LANG/checkpoint-final/pytorch_model.bin \
       $OUT_DIR/checkpoint

    cp $CCT5_OUTPUT/test_${LANG}_random-preds.txt \
       $OUT_DIR/predictions/mcmd-nl_preds.txt

    cp $CCT5_OUTPUT/test_${LANG}_random-golds.txt \
       $OUT_DIR/predictions/mcmd-nl_golds.txt

    echo "Evaluating CCT5 MCMD-NL - $LANG"

    python $EVAL_SCRIPT \
        --gold $OUT_DIR/predictions/mcmd-nl_golds.txt \
        --pred $OUT_DIR/predictions/mcmd-nl_preds.txt \
        > $OUT_DIR/metrics_mcmd-nl.txt

    ########################################
    # ========== COME ======================
    ########################################

    source /media/apps/avx512-2021/software/Anaconda3/2025.06-1/etc/profile.d/conda.sh
    conda activate come_pt2

    cd $COME_DIR || exit 1

    echo "Running COME MCMD-NL - $LANG"

    ########################################
    # RUN ID STRUCTURE
    ########################################

    RUN_ID="EXP2_LS1_RUN2_mcmd-nl_${LANG}"

    OUT_DIR=$BASE_EXP/COME/$LANG

    EXP_OUT=$COME_DIR/output/$RUN_ID
    EXP_CACHE=$COME_DIR/cache/$RUN_ID
    EXP_SUMMARY=$COME_DIR/summary/$RUN_ID
    EXP_RES=$COME_DIR/res/$RUN_ID

    mkdir -p \
      $OUT_DIR/predictions \
      $EXP_OUT \
      $EXP_CACHE \
      $EXP_SUMMARY \
      $EXP_RES

    ########################################
    # TRAINING MCMD-NL
    ########################################

    echo "Training COME MCMD-NL - $LANG"

    python run_gen.py \
        --task mcmd-nl \
        --sub_task $LANG \
        --data_dir data/data \
        --train_filename $LANG/train.jsonl \
        --dev_filename $LANG/valid.jsonl \
        --test_filename $LANG/test.jsonl \
        --do_train \
        --do_eval \
        --model_name_or_path Salesforce/codet5-base \
        --tokenizer_name Salesforce/codet5-base \
        --output_dir $EXP_OUT \
        --cache_path $EXP_CACHE \
        --summary_dir $EXP_SUMMARY \
        --res_dir $EXP_RES \
        --add_task_prefix \
        --always_save_model

    ########################################
    # INFERENCE MCMD-NL
    ########################################

    echo "Inference COME MCMD-NL - $LANG"

    python run_gen.py \
        --task mcmd-nl \
        --sub_task $LANG \
        --data_dir data/data \
        --test_filename $LANG/test.jsonl \
        --do_test \
        --model_name_or_path Salesforce/codet5-base \
        --tokenizer_name Salesforce/codet5-base \
        --load_model_path $EXP_OUT/checkpoint-best-ppl/pytorch_model.bin \
        --output_dir $EXP_OUT \
        --cache_path $EXP_CACHE \
        --summary_dir $EXP_SUMMARY \
        --res_dir $EXP_RES \
        --add_task_prefix

    COME_RES=$EXP_RES

    ########################################
    # COPY RESULTS
    ########################################

    # Copy the checkpoint to the output directory for record-keeping
    cp $EXP_OUT/checkpoint-best-ppl/pytorch_model.bin \
       $OUT_DIR/checkpoint

    cp $COME_RES/test_best-ppl.output \
       $OUT_DIR/predictions/mcmd-nl_preds.txt

    cp $COME_RES/test_best-ppl.gold \
       $OUT_DIR/predictions/mcmd-nl_golds.txt

    ########################################
    # EVALUATION
    ########################################

    echo "Evaluating COME MCMD-NL - $LANG"

    python $EVAL_SCRIPT \
        --gold $OUT_DIR/predictions/mcmd-nl_golds.txt \
        --pred $OUT_DIR/predictions/mcmd-nl_preds.txt \
        > $OUT_DIR/metrics_mcmd-nl.txt

    echo "DONE: MCMD-NL finished for $LANG"

done

echo "RUN COMPLETE"