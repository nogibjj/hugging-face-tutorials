[![CI](https://github.com/nogibjj/mlops-template/actions/workflows/cicd.yml/badge.svg?branch=GPU)](https://github.com/nogibjj/mlops-template/actions/workflows/cicd.yml)
[![Codespaces Prebuilds](https://github.com/nogibjj/mlops-template/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg?branch=GPU)](https://github.com/nogibjj/mlops-template/actions/workflows/codespaces/create_codespaces_prebuilds)

## Hugging Face Tutorials

### Create data

Manually upload data from UX or from API.

* [Create your own dataset Reference](https://huggingface.co/course/chapter5/5?fw=pt)

To load do the following:

```
from datasets import load_dataset
remote_dataset = load_dataset("noahgift/social-power-nba")
remote_dataset
```
#### Recommended Tutorial Followup

1.  Find a simple and small dataset: kaggle, your own, a sample dataset
2.  Go to Hugging Face website and upload
3.  Download and explore dataset
4.  Enhance dataset by filling out dataset metadata.
5.  Build a Demo for it.


### Create model

* [Sharing a model](https://huggingface.co/course/chapter4/3?fw=pt)
* [Creating a model card](https://huggingface.co/course/chapter4/4?fw=pt)

#### Recommended Tutorial Followup

1. Upload model to Hugging Face website
2. Fill out model card
3. Use model

### Fine-Tuning Hugging Face Models Tutorial

Why transfer learning?

![10-7-transformers](https://user-images.githubusercontent.com/58792/196711699-8034d017-a2bb-4ec3-8029-04c925cbf254.png)

* One batch in PyTorch
* Using sacrebleu (precision based "Precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances").  Recall is "while recall (also known as sensitivity) is the fraction of relevant instances that were retrieved" - wikipedia
* The ROUGE score was specifically developed for applications like summarization where high recall is more important than just precision! 
```python
rouge_metric = load_metric("rouge")
```


```python
from datasets import load_metric
bleu_metric = load_metric("sacrebleu")
```




### Push to Hub

* Need token and [follow guide](https://huggingface.co/docs/transformers/model_sharing)



* [Refer to HuggingFace course](https://huggingface.co/course/chapter3/2?fw=pt)



### Create spaces




### Verify GPU works

The following examples test out the GPU

* run pytorch training test: `python utils/quickstart_pytorch.py`
* run pytorch CUDA test: `python utils/verify_cuda_pytorch.py`
* run tensorflow training test: `python utils/quickstart_tf2.py`
* run nvidia monitoring test: `nvidia-smi -l 1` it should show a GPU
* run whisper transcribe test `./utils/transcribe-whisper.sh` and verify GPU is working with `nvidia-smi -l 1`

Additionally, this workspace is setup to fine-tune Hugging Face

![fine-tune](https://user-images.githubusercontent.com/58792/195709866-121f994e-3531-493b-99af-c3266c4e28ea.jpg)


`python hf_fine_tune_hello_world.py` 

### Used in Following Projects

Used as the base and customized in the following Duke MLOps and Applied Data Engineering Coursera Labs:

* [MLOPs-C2-Lab1-CICD](https://github.com/nogibjj/Coursera-MLOPs-Foundations-Lab-1-CICD)
* [MLOps-C2-Lab2-PokerSimulator](https://github.com/nogibjj/Coursera-MLOPs-Foundations-Lab-2-poker-simulator)
* [MLOps-C2-Final-HuggingFace](https://github.com/nogibjj/Coursera-MLOps-C2-Final-HuggingFace)
* [Coursera-MLOps-C2-lab3-probability-simulations](Coursera-MLOps-C2-lab3-probability-simulations)
* [Coursera-MLOps-C2-lab4-greedy-optimization](https://github.com/nogibjj/Coursera-MLOps-C2-lab4-greedy-optimization)

### References

* [nlp-with-transformers
/
notebooks
](https://github.com/nlp-with-transformers/notebooks)
* [Natural Language Processing with Transformers, Revised Edition](https://learning.oreilly.com/library/view/natural-language-processing/9781098136789/)
* [Building Cloud Computing Solutions at Scale Specialization](https://www.coursera.org/specializations/building-cloud-computing-solutions-at-scale)
* [Python, Bash and SQL Essentials for Data Engineering Specialization](https://www.coursera.org/learn/web-app-command-line-tools-for-data-engineering-duke)
* [Implementing MLOps in the Enterprise](https://learning.oreilly.com/library/view/implementing-mlops-in/9781098136574/)
* [Practical MLOps: Operationalizing Machine Learning Models](https://www.amazon.com/Practical-MLOps-Operationalizing-Machine-Learning/dp/1098103017)
* [Coursera-Dockerfile](https://gist.github.com/noahgift/82a34d56f0a8f347865baaa685d5e98d)
