# Sentiment Analysis: Embedding vs. LLM Classifier

This repository presents an experiment comparing two approaches for sentiment analysis on financial news tweets.

### Approaches

1. **ModernBERT with Classification Head**  
   - Fine-tunes a pretrained ModernBERT model using a standard classification head for sentiment analysis.

2. **Qwen-based LLM Classifier**  
   - Adapts a causal language model (Qwen2.5-0.5B) into a classifier by masking out (i.e., setting to `-float('inf')`) all output logits except those corresponding to the target sentiment labels.

3. **ModernBert with [MASK] token**
   - Uses ModernBert as a generative model (https://arxiv.org/pdf/2502.03793)

### Dataset

- **Source:** [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)  
- **Details:** Contains tweets labeled with financial sentiments (e.g., bearish, bullish, neutral).

### Setup

Install the required packages using:

```bash
pip install transformers datasets torch scikit-learn
```

### Results


