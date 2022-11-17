# Masked Language Model

## Train tokenizer

It only took 5 minutes to read whole dataset and train tokenizer.

## Processing dataset

Code to process dataset has ~170GB peak storage usage, but the final size is only 81GB after cache is cleared. Reducing peak storage usage **should be** possible if cache is cleared halfway, but i didn't test it. It took 3-4 hours to process the dataset with total 822986 train row and 43316 evaluation row.
