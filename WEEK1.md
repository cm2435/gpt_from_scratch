# WEEK 1: Implementing Self-Attention

## Project Setup

To get started with the project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/cm2435/gpt_from_scratch
    cd gpt_from_scratch
    ```

2. **Set up the Python environment using Poetry**:
    ```bash
    poetry install
    poetry shell
    ```

## Repository Structure

Here is an overview of the repository structure:



### Important Files and Directories

- **`src/model/attention/impl.py`**: This is where you will implement the self-attention mechanism. There are other files, we will cover them in later weeks of the course. 

- **`tests/reference_implimentations`**: Contains reference implementations for you to refer to if you get stuck.
    - **`reference_causal_attention.py`**: Reference implementation of the causal self-attention mechanism.

- **`tests/test_attention.py`**: Contains tests for your self-attention implementation. These are implimented with unitest. 

## Getting Started with Week 1

This week, your task is to implement the self-attention mechanism in the `src/model/attention/impl.py` file. Follow these steps:

1. **Open `src/model/attention/impl.py`** and fill in the blanks to complete the self-attention implementation.
2. Refer to the **`tests/reference_implimentations/reference_causal_attention.py`** file if you need guidance or get stuck.
3. Run the tests in **`tests/test_attention.py`** to verify your implementation.


### Step-by-Step Instructions
Here is a step by step guide of how to tackle our implimentation of causal multi head attention!
1. **Generate queries, keys, and values for all heads**
    ```python
    #1. Generate queries, keys, and values for all heads
    ```
    - **Question**: How do we get the output of the input sequence multiplied by the key matrix? You might find help [here](https://jalammar.github.io/illustrated-transformer/).
    - **Hint**: It's common syntax to get the output of a torch layer by doing something like
    ```python
    import torch.nn as nn 

    class Layer(nn):
        def __init__(self):
            self.transformer = nn.Linear(input_dim, output_dim)
    
        def transform(x: torch.tensor) -> torch.tensor:
            return self.transformer(x)
    ```

2. **Reshape and transpose to separate heads and prepare for multi-head attention**
    - **Question**: How do we reshape and transpose tensors in PyTorch? 
    This [tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) may help.
    Also the [API Reference for tensor Views](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html)
    Also the [API Reference for tensor reshaping](https://pytorch.org/docs/stable/generated/torch.reshape.html)

3. **Reshape again to merge the batch size and number of heads into a single dimension**
    - **Question**: Why is it important to merge batch size and number of heads for matrix multiplication? Check this [illustration](https://jalammar.github.io/illustrated-transformer/).
    - **Hint**: Really consider the shape of your matrices, and what the self attention operation would be for a single matrix, and then abstract it to a batch. Read the comments for shape suggestions!

4. **Perform batch matrix multiplication to compute attention scores**
    - **Question**: How do we use `torch.bmm` for batch matrix multiplication? More details [here](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).
    Also the [API Reference for torch bmm](https://pytorch.org/docs/stable/generated/torch.bmm.html)

5. **Apply causal mask by setting future positions to negative infinity**
    - **Question**: Why is a causal mask necessary? Check this [resource](https://jalammar.github.io/illustrated-gpt2/).

6. **Apply softmax to normalize the scores**
    - **Question**: What is the softmax function? How do we impliment it? You can find the pytorch reference [here](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)
    - **Question**: How does softmax help in normalizing attention scores? You might find more information [here](https://jalammar.github.io/illustrated-transformer/).

7. **Find the Attention output**
    - **Question**: What is the self attention equation? Is it different for batched inputs? How do we compute the final attention output? Refer to [this guide](https://jalammar.github.io/illustrated-gpt2/).

8. **Concatenate the outputs of different heads and project**
    - **Question**: Why do we need to concatenate the outputs of different heads? This [tutorial](https://jalammar.github.io/illustrated-transformer/) can help understand Multi Head attention.


9. **Running the tests** Finally, run the tests in **`tests/test_attention.py`** with the below command 
```bash
python -m tests.test_attention --v
```
 to verify your implementation! The call to 
```python
torch.assert.all_close
```
Will test the numerical accuracy of your precision as well as if the shaping is correct! 


- **Understanding Transformers**: 
  - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  - [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

- **PyTorch Basics**:
  - [PyTorch: 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

Happy coding! If you have any questions or need further assistance, feel free to reach out.
