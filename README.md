# Conceptual Cultural Index: A Metric for Cultural Specificity via Relative Generality
[![arXiv](https://img.shields.io/badge/arXiv-2602.09444-b31b1b.svg)](https://arxiv.org/abs/2602.09444)

Author: Ohashi Takumi, Iyatomi Hitoshi

Abstract: Large language models (LLMs) are increasingly deployed in multicultural settings; however, systematic evaluation of cultural specificity at the sentence level remains underexplored. We propose the Conceptual Cultural Index (CCI), which estimates cultural specificity at the sentence level. CCI is defined as the difference between the generality estimate within the target culture and the average generality estimate across other cultures. This formulation enables users to operationally control the scope of culture via comparison settings and provides interpretability, since the score derives from the underlying generality estimates. We validate CCI on 400 sentences (200 culture-specific and 200 general), and the resulting score distribution exhibits the anticipated pattern: higher for culture-specific sentences and lower for general ones. For binary separability, CCI outperforms direct LLM scoring, yielding more than a 10-point improvement in AUC for models specialized to the target culture. 


## Usage
### Installation
``` bash
$ git clone git@github.com:IyatomiLab/CCI.git
$ cd CCI

$ uv sync
```

### 1) Compute CCI (Global mode)
```
uv run python run_cci.py \
  --text "Taking milk out of the refrigerator." \
  --model openai/gpt-oss-20b \
  --target-culture "United States of America"
```
**Example Output**
```
CCI: 0.0417
Generality Scores: {'Argentina': 0.950, 'Australia': 0.950, 'Brazil': 0.933, 'Canada': 0.950, 'China': 0.850, 'France': 0.950, 'Germany': 0.950, 'India': 0.800, 'Indonesia': 0.783, 'Italy': 0.950, 'Japan': 0.917, 'Mexico': 0.933, 'Republic of Korea': 0.917, 'Republic of South Africa': 0.900, 'Russian Federation': 0.917, 'Saudi Arabia': 0.850, 'Turkey': 0.900, 'United Kingdom': 0.950, 'United States of America': 0.950}
```

### 2) Compute CCI (Custom mode)
```
uv run python run_cci.py \
  --text "節分に豆を撒く。" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --cultures "China" "Republic of Korea" "United States of America" "Japan" \
  --target-culture "Japan"
```
**Example Output**
```
CCI: 0.8333
Generality Scores: {'China': 0.200, 'Republic of Korea': 0.100, 'United States of America': 0.050, 'Japan': 0.950}
```


## Citation
``` bibtex
@article{ohashi2026cci,
  author = {Ohashi, Takumi and Iyatomi, Hitoshi},
  title = {Conceptual Cultural Index: A Metric for Cultural Specificity via Relative Generality},
  journal = {arXiv preprint arXiv:2602.09444},
  year = {2026}
}
```
