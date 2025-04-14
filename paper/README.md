# LLMs Sensitivity & Consistency (NAACL 2025)

This is the official repository of the [NAACL 2025 paper](https://arxiv.org/abs/2406.12334) _"What Did I Do Wrong? Quantifying LLMs’ Sensitivity and Consistency to Prompt Engineering"_.

### Citing our work

If you found our metrics useful, please cite our work:
```
@inproceedings{errica_what_2025,
  author    = {Federico Errica and
               Giuseppe Siracusano and
               Davide Sanvito and
               Roberto Bifulco},
  title     = {What Did I Do Wrong? Quantifying LLMs’ Sensitivity and Consistency to Prompt Engineering},
  booktitle = {Proceedings of the 2025 Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL)},
  year      = {2025},
}
```

## Installation

After you have cloned the repository, let's create an ad-hoc environment for this project

```bash
python3.10 -m venv .venv/llm
source .venv/llm/bin/activate
pip install -r requirements.txt
```
Now you need to update the `constants.py` file with the URL of your LLama3 and Mixtral servers. 
Please refer to `utils.py` to see how this information is used in classes `DefaultNLELlama3_70bChatOpenAI` and `DefaultNLEMixtral_8x7bChatOpenAI`.

## Run the Notebooks
You should now be able to run our notebooks to reproduce our results. There is one notebook for each dataset tested.
