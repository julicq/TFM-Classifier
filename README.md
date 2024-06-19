# IMDb Sentiment Classification with DistilBERT

This project uses the DistilBERT model to perform sentiment classification on the IMDb dataset. The script fine-tunes the pre-trained DistilBERT model using the IMDb dataset and evaluates its performance.

## Setup

### Prerequisites

- Python 3.8 or later
- An NVIDIA GPU with CUDA installed (optional, but recommended for faster training)

### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-repository/imdb-sentiment-classification.git
    cd imdb-sentiment-classification
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training and Evaluation

To train and evaluate the model, run the following command:

```sh
python tfm_classifier.py
```

This script will:

- Load and preprocess the IMDb dataset.
- Fine-tune the DistilBERT model.
- Evaluate the model on the test set.
- Save the fine-tuned model and tokenizer.

## Requirements

- *torch:* PyTorch for model training and inference.
- *transformers:* Hugging Face Transformers library for using the DistilBERT model.
- *datasets:* Hugging Face Datasets library for loading and processing the IMDb dataset.
- *pandas:* Data manipulation library.


## Notes:

- Replace `your-repository` with the actual repository URL if you have one.
- Make sure the `tfm_classifier.py` script contains the training and evaluation code provided earlier.

This `README.md` provides comprehensive instructions on setting up, running, and using the project, ensuring clarity for any users or contributors.

## License
This project is licensed under the MIT License. See the *LICENSE* file for details.