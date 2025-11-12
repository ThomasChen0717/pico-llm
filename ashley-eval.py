import torch
import torch.nn.functional as F
from collections import Counter
from pico-llm import compute_next_token_loss, generate_text


def compute_perplexity(model, data_loader, device):
    """Compute perplexity on a dataset"""

    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Loop through all batches in the dataset
        for batch in data_loader:
            # Move batch to correct device (CPU or GPU)
            batch = batch.to(device)

            # Forward pass: get model predictions
            logits = model(batch)

            # Compute cross-entropy loss for next token prediction
            loss = compute_next_token_loss(logits, batch)

            # Count non-padding tokens (assuming 0 is padding)
            n_tokens = (batch[1:] != 0).sum().item()

            # Accumulate weighted loss
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    # Compute average loss per token
    avg_loss = total_loss / total_tokens

    # Perplexity = exp(average loss)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


def compute_diversity(generated_texts, n=2):
    """
    Compute diversity as ratio of unique n-grams to total n-grams.

    Args:
        generated_texts: list of generated strings
        n: n-gram size (1=unigram, 2=bigram, 3=trigram)

    Returns:
        diversity: float between 0 and 1
    """
    all_ngrams = []

    # Extract all n-grams from all generated texts
    for text in generated_texts:
        # Split text into words
        words = text.split()

        # Extract n-grams using sliding window
        for i in range(len(words) - n + 1):
            # Create n-gram as tuple of n consecutive words
            ngram = tuple(words[i:i + n])
            all_ngrams.append(ngram)

    # Handle edge case: no n-grams found
    if len(all_ngrams) == 0:
        return 0.0

    # Count unique n-grams (set removes duplicates)
    unique_ngrams = len(set(all_ngrams))

    # Count total n-grams
    total_ngrams = len(all_ngrams)

    # Diversity = proportion of unique n-grams
    # Higher diversity means more varied/creative text
    diversity = unique_ngrams / total_ngrams

    return diversity


def compare_sampling_methods(model, enc, prompt, device, n_samples=20):
    """Generate samples with different methods and compare"""
    # Define different sampling strategies to test
    methods = {
        'greedy': {'top_p': None},  # Always pick most likely token
        'nucleus_0.9': {'top_p': 0.9},  # Sample from top 90% probability mass
        'nucleus_0.95': {'top_p': 0.95},  # Sample from top 95% probability mass
        'nucleus_1.0': {'top_p': 1.0},  # Sample from full distribution
    }

    results = {}

    # Test each sampling method
    for method_name, kwargs in methods.items():
        print(f"Generating with {method_name}...")
        samples = []

        # Generate n_samples texts with this method
        for i in range(n_samples):
            # Generate text using specified sampling method
            text, _ = generate_text(model, enc, prompt, max_new_tokens=30,
                                    device=device, **kwargs)
            samples.append(text)

        # Compute diversity metrics
        # Unigram diversity: how many unique words
        diversity_1 = compute_diversity(samples, n=1)

        # Bigram diversity: how many unique 2-word phrases
        diversity_2 = compute_diversity(samples, n=2)

        # Store results for this method
        results[method_name] = {
            'samples': samples,  # All generated texts
            'diversity_unigram': diversity_1,  # Word-level diversity
            'diversity_bigram': diversity_2  # Phrase-level diversity
        }

    return results