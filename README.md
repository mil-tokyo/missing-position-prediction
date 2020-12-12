# Finding and Generating a Missing Part for Story Completion

This repository will contain the code of _Finding and Generating a Missing Part for Story Completion_ by Yusuke Mori, Hiroaki Yamane, Yusuke Mukuta and Tatsuya Harada (LaTeCH-CLfL 2020). [PDF](https://www.aclweb.org/anthology/2020.latechclfl-1.19.pdf)

## dependencies
We used Python 3.6 for implementation. 

The required external libraries are as follows:

- jupyterlab
- pytorch >= v1.4.0
- numpy
- pandas
- tqdm
- spacy
- transformers (version 2 (or 3) is recommended. It seems some modification should be needed for the version 4 or later.)
- sentence-transformers
- tensorboard
- mlflow
- matplotlib
- seaborn

## dataset

Please get the ROCStories dataset from https://cs.rochester.edu/nlp/rocstories/ 

"ROCStories_preprocess.ipynb" will help you to split the dataset as we have done for our task.

Note that we did not set `random_state` for `sklearn.model_selection.train_test_split`.  
This may cause the reproduction result of the experiment somewhat different from those in our paper, but this does not affect the main claims of the paper.  
(The dataset requires registration to obtain it, so it will not be published here. Sorry.)

## reproduction

### Experiment 1

See "MissingPositionPrediction" directory.

`sentencebert_missingpositionprediction.py` is the main source code.

For GRU Context, please run the code as below.
```
$ CUDA_VISIBLE_DEVICES=0 python sentencebert_missingpositionprediction.py --output_dir gru_seed42 --seed 42 --epochs 30
```

For Max-pool Context, please run the code as below.
```
$ CUDA_VISIBLE_DEVICES=0 python sentencebert_missingpositionprediction.py --output_dir pool_seed42 --seed 42 --epochs 30 -ce PoolContextEncoder
```

The five random seeds we used were 42, 1234, 99, 613, and 1000.

Note: unfortunately, the class name is misspelled (Missing P'i'sition), but we have not corrected them at this time so that the trained model can be re-used. 

#### Table 2

- mpp_test_set_result_gru.ipynb
- mpp_test_set_result_maxpool.ipynb
 
#### Figure 3

- mpp_heatmap_load_calculated.ipynb

You may instead use files below.
- missing_position_prediction_show_result_gru.ipynb
- missing_position_prediction_show_result_maxpool.ipynb

## Experiment 2

See "MPP_StoryCompletion" directory.

`sbert_context_bert_multitask_storycompletion.py` is the main source code.

```
$ CUDA_VISIBLE_DEVICES=0 python sbert_context_bert_multitask_storycompletion.py --output_dir GRUContext_batch128_lr00001_50epoch --seed 42 --epochs 50 --warmup_epochs 4 --batch-size 128 --learning_rate 1e-4
```

### Human Evaluation

Amazon MTurk result is in `AmazonMTurk` directory.

You can reproduce Table 3 with `result_count.ipynb`.

The questions we used can be reproduced by `create_question_csv.ipynb` (we used index 0 - 199 of the created questions).

## Citation

If you find this repository helpful for your work, please consider citing the related paper.

> Yusuke Mori, Hiroaki Yamane, Yusuke Mukuta, and Tatsuya Harada. 2020. Finding  and  Generating a Missing Part for Story Completion. In Proceedings of the The 4th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, SocialSciences, Humanities and Literature, pages 156–166, Online, December. International Committee on Computational Linguistics.

Please visit the ACL Anthology to get [Bibtex file](https://www.aclweb.org/anthology/2020.latechclfl-1.19.bib).

## Contact

If you have any inquiries (find something strange in this repository, etc.), please feel free to open an issue in this repository or send email to Yusuke Mori (corresponding author).

E-mail address: mori at mi.t.u-tokyo.ac.jp