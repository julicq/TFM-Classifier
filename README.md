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
python train.py
