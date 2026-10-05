### Deep Learning Systems Project

##### Project Description

This project implements a GPT-style decoder-only Transformer from first principles and applies it to conversational language modeling using the DailyDialog dataset. Using Python, PyTorch, NumPy, Matplotlib, and the Hugging Face Transformers library, the project demonstrates an end-to-end deep learning workflow including dataset exploration, tokenization, Transformer implementation, model training, architectural experimentation, and evaluation.

The primary objective of the project is to train a language model capable of predicting the next token in a conversation given all preceding context. A baseline Transformer architecture was developed and evaluated before conducting controlled experiments investigating the effects of increased Transformer depth and embedding dimensionality on model performance, generalization, and parameter efficiency.

##### What I Built

- Loaded, validated, and explored the DailyDialog conversational dataset.
- Performed exploratory analysis of dialogue lengths, vocabulary utilization, and conversational structure.
- Applied GPT-2 tokenization.
- Implemented token embeddings and positional embeddings.
- Implemented causal multi-head self-attention with masking.
- Developed Transformer blocks using layer normalization, feed-forward networks, residual connections, and dropout.
- Built a GPT-style decoder-only Transformer architecture for conversational language modeling.
- Developed training, validation, and testing pipelines using PyTorch.
- Implemented AdamW optimization, checkpointing, and training observability through loss tracking and generation checkpoints.
- Conducted controlled architectural experiments involving increased Transformer depth and increased embedding dimensionality.
- Evaluated model performance using training, validation, and test loss metrics.
- Analyzed architectural trade-offs involving model capacity, parameter efficiency, and generalization performance.
- Summarized findings, limitations, ethical considerations, and future enhancement opportunities.

##### Dataset

**DailyDialog**

Source: [Hugging Face](https://huggingface.co/datasets/roskoN/dailydialog)

Paper: DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset (Li et al., 2017)

**Files Used**

```Text
train/dialogues_train.txt
validation/dialogues_validation.txt
test/dialogues_test.txt
```

**Dataset Characteristics**

The DailyDialog corpus contains approximately 11,118 manually curated multi-turn conversations covering common real-world topics such as:

- Relationships
- Education
- Work
- Health
- Finance
- Culture
- Daily life activities

The dataset provides predefined training, validation, and testing splits, making it well suited for language modeling and controlled experimental evaluation.

##### Notebook Execution Notes

This project was developed locally and trained using Google Colab GPU resources.

A small number of Colab-specific helper commands remain in the notebook as comments for reproducibility, for example:

```Python
# %cd /content/deep-learning-systems-project/notebooks
# !ls -la
```

These commands are not required when running the notebook locally and may be safely ignored.

Transformer training experiments were executed using GPU-enabled Colab runtimes due to the computational requirements of training decoder-only language models. All results, analysis, experiments, and discussion are included within the notebook and accompanying report, so reviewers are not required to retrain the models to evaluate the project.

##### Bias Awareness and Limitations

As with all language modeling projects, the results should be interpreted with appropriate caution.

**Potential limitations include:**

The DailyDialog dataset is substantially smaller than the large-scale corpora typically used to train modern language models.
Conversational content may not fully represent the diversity of real-world language use.
Generated responses may appear fluent while still containing factual inaccuracies or semantic inconsistencies.
Computational constraints influenced model size, context length, and training duration.
Evaluation was primarily based on loss metrics and qualitative generation examples rather than large-scale human evaluation studies.

##### Future Extensions

Several opportunities exist to extend this project and further improve conversational language modeling performance.

`Architectural Improvements`

Future experiments could investigate:

- Larger context lengths
- Deeper Transformer architectures
- Alternative attention mechanisms
- Different regularization strategies
- Longer training schedules
- Tokenization Research

An interesting extension would be the development of a custom tokenizer specifically designed for the DailyDialog corpus.

During exploratory analysis, only a subset of the GPT-2 vocabulary was utilized by the dataset, suggesting that a smaller domain-specific vocabulary may improve token efficiency while reducing model complexity.

##### How to Run the Project

**Model Checkpoints are not saved in Github or Git because of size constraints**

1. Clone the Repository

```Shell
git clone <repository-url>
cd deep-learning-systems-project
```

2. Create and Activate an Environment

Using Conda:

```Shell
conda env create -f environment.yml
conda activate statistical_analysis
```

Using venv:

```Shell
python -m venv statistical_analysis
```

Windows:

```Shell
statistical_analysis\Scripts\activate.bat
```

Linux/macOS:

```Shell
source statistical_analysis/bin/activate
```

3. Install Dependencies

Python Version

`Python 3.11+`

Using pip:

```Shell
pip install -r requirements.txt
```

Using Conda:

```Shell
conda env create -f environment.yml
```

4. Open the Notebook

Launch Jupyter:

```Shell
jupyter notebook
```

Open:

```Text
deep_learning_systems_project.ipynb
```

Run all cells from top to bottom.

Running with Google Colab

The notebook was designed to support both local execution and GPU-enabled execution within Google Colab.

Typical workflow:

- Clone the repository within Colab.
- Verify correct Paths using Terminal or Notebook Bash commands.
- Enable a GPU runtime.
- Install project dependencies.
- Run notebook cells sequentially.
- Execute model training experiments.

Reviewers may execute the notebook locally without requiring Colab-specific configuration.

##### Reproducibility

Generate the requirements file:

```Shell
pip freeze > requirements.txt
```

Generate the Conda environment specification:

```Shell
conda env export > environment.yml
```

The requirements.txt and environment.yml files are included in this repository to support reproducibility and environment recreation.
