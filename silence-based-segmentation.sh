python silence-based-segmentation.py \
--hf_token "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"  \
--dataset_name ArissBandoss/proverbes-moore-vol2 \
--output_folder segmented-data-silence \
--output_dataset segmented_dataset.json \
--silence_thresh -40 \
--min_silence_len 700