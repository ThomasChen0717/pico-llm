# starter code by matus & o1-pro
import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")

    args = parser.parse_args()
    return args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        # fill in

        self.net = None

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        """
        seq_len, batch_size = tokens_seq.shape
        outputs = []

        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            for t in range(start, end):
                batch_logits = []
                for b in range(batch_size):
                    if t < self.k:
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    context_oh = F.one_hot(
                        torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device),
                        num_classes=self.vocab_size
                    )
                    context_flat = context_oh.flatten().float().unsqueeze(0)
                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
        return outputs


################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024, 
                 num_layers=1, dropout=0.0, bidirectional=False):
        """
        Args:
            vocab_size (int): Size of vocabulary
            embed_size (int): Embedding dimension
            hidden_size (int): LSTM hidden dimension
            num_layers (int): Number of stacked LSTM layers (default: 1)
            dropout (float): Dropout between LSTM layers (default: 0.0)
            bidirectional (bool): Bidirectional LSTM (default: False)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Multi-layer LSTM with optional dropout and bidirectional
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  # Dropout only works with > 1 layer
            bidirectional=bidirectional,
            batch_first=False
        )
        
        # Output layer
        # If bidirectional, LSTM output size is doubled
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.linear = nn.Linear(lstm_output_size, vocab_size)
    
    def forward(self, tokens_seq):
        """
        Forward pass
        
        Args:
            tokens_seq: (seq_len, batch) - Input token IDs
        
        Returns:
            logits: (seq_len, batch, vocab_size) - Output logits
        """
        # Embedding: (seq_len, batch) -> (seq_len, batch, embed_size)
        emb = self.embedding(tokens_seq)
        
        # LSTM: (seq_len, batch, embed_size) -> (seq_len, batch, hidden_size * num_directions)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)
        
        # Linear: (seq_len, batch, hidden * directions) -> (seq_len, batch, vocab_size)
        logits = self.linear(out)
        
        return logits

def create_train_val_split(dataset, val_ratio=0.1, seed=42):
    """
    Split dataset into training and validation sets
    
    Args:
        dataset: torch.utils.data.Dataset
        val_ratio (float): Ratio of validation data (default: 0.1)
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_dataset, val_dataset)
    
    Example:
        >>> train_data, val_data = create_train_val_split(dataset, val_ratio=0.1)
        >>> print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    """
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    # Split with fixed seed for reproducibility
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Dataset split: {train_size} train, {val_size} val")
    
    return train_dataset, val_dataset


def setup_optimizer_and_scheduler(model, lr=1e-3, patience=3, factor=0.5, min_lr=1e-6):
    """
    Setup Adam optimizer and ReduceLROnPlateau scheduler
    
    The scheduler reduces learning rate when validation loss plateaus
    
    Args:
        model: Neural network model
        lr (float): Initial learning rate (default: 1e-3)
        patience (int): Epochs with no improvement before LR reduction (default: 3)
        factor (float): Factor to multiply LR by when reducing (default: 0.5)
        min_lr (float): Minimum learning rate (default: 1e-6)
    
    Returns:
        tuple: (optimizer, scheduler)
    
    Example:
        >>> optimizer, scheduler = setup_optimizer_and_scheduler(model)
        >>> # After each epoch:
        >>> scheduler.step(val_loss)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # Minimize validation loss
        factor=factor,        # New LR = LR * factor
        patience=patience,    # Wait this many epochs before reducing
        min_lr=min_lr,        # Don't go below this LR
        verbose=True          # Print when LR is reduced
    )
    
    return optimizer, scheduler


def clip_gradients(model, max_norm=1.0):
    """
    Clip gradients to prevent exploding gradients
    
    This is especially important for RNNs/LSTMs which can have
    gradient explosion issues during backpropagation through time
    
    Args:
        model: Neural network model
        max_norm (float): Maximum gradient norm (default: 1.0)
    
    Returns:
        float: Total gradient norm before clipping
    
    Example:
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> grad_norm = clip_gradients(model, max_norm=1.0)
        >>> optimizer.step()
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm.item()


def validate_model(model, val_loader, device, compute_next_token_loss):
    """
    Evaluate model on validation set
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        device: torch device
        compute_next_token_loss: Loss function from pico-llm.py
    
    Returns:
        float: Average validation loss
    
    Example:
        >>> val_loss = validate_model(model, val_loader, device, compute_next_token_loss)
        >>> print(f"Validation loss: {val_loss:.4f}")
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_tokens in val_loader:
            batch_tokens = batch_tokens.to(device)
            
            # Forward pass
            logits = model(batch_tokens)
            loss = compute_next_token_loss(logits, batch_tokens)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


# ============================================================================
# METRICS TRACKER
# ============================================================================

class MetricsTracker:
    """
    Track and visualize training metrics over epochs
    
    Example:
        >>> tracker = MetricsTracker()
        >>> for epoch in range(num_epochs):
        >>>     # ... training code ...
        >>>     tracker.add_epoch(epoch, train_loss, val_loss, lr)
        >>> tracker.plot_learning_curves('my_curves.png')
        >>> gap = tracker.get_overfitting_gap()
    """
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs = []
    
    def add_epoch(self, epoch, train_loss, val_loss, lr):
        """
        Add metrics for one epoch
        
        Args:
            epoch (int): Epoch number
            train_loss (float): Training loss
            val_loss (float): Validation loss
            lr (float): Current learning rate
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
    
    def get_overfitting_gap(self):
        """
        Calculate average gap between validation and training loss
        
        Returns:
            float: Average (val_loss - train_loss)
        """
        if len(self.train_losses) == 0:
            return 0.0
        
        gaps = [val - train for train, val in zip(self.train_losses, self.val_losses)]
        return sum(gaps) / len(gaps)
    
    def get_final_metrics(self):
        """
        Get final epoch metrics
        
        Returns:
            dict: Final metrics
        """
        if len(self.epochs) == 0:
            return {}
        
        return {
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_lr': self.learning_rates[-1],
            'best_val_loss': min(self.val_losses),
            'overfitting_gap': self.get_overfitting_gap()
        }
    
    def plot_learning_curves(self, save_path='learning_curves.png'):
        """
        Plot training curves and save to file
        
        Creates a 2-panel figure:
        - Left: Train vs Validation loss
        - Right: Learning rate schedule
        
        Args:
            save_path (str): Path to save the figure
        """
        if len(self.epochs) == 0:
            print("No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left panel: Loss curves
        ax1.plot(self.epochs, self.train_losses, 'b-o', 
                label='Train Loss', linewidth=2, markersize=5, alpha=0.7)
        ax1.plot(self.epochs, self.val_losses, 'r-o', 
                label='Val Loss', linewidth=2, markersize=5, alpha=0.7)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(alpha=0.3)
        
        # Add gap annotation
        final_gap = self.val_losses[-1] - self.train_losses[-1]
        ax1.annotate(f'Final Gap: {final_gap:.3f}',
                    xy=(self.epochs[-1], self.val_losses[-1]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                    fontsize=10)
        
        # Right panel: Learning rate
        ax2.plot(self.epochs, self.learning_rates, 'g-o', 
                linewidth=2, markersize=5, alpha=0.7)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')  # Log scale for LR
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Learning curves saved: {save_path}")
        plt.close()


# ============================================================================
# ENHANCED TRAINING LOOP
# ============================================================================

def train_lstm_with_validation(
    model, 
    train_loader, 
    val_loader,
    epochs, 
    device,
    lr=1e-3,
    max_grad_norm=1.0,
    log_steps=100,
    model_name='lstm',
    compute_next_token_loss=None
):
    """
    Train LSTM with validation, gradient clipping, and LR scheduling
    
    This is an enhanced version of the training loop that includes:
    - Validation set evaluation
    - Learning rate scheduling
    - Gradient clipping
    - Metrics tracking
    
    Args:
        model: LSTM model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs (int): Number of training epochs
        device: torch device
        lr (float): Initial learning rate (default: 1e-3)
        max_grad_norm (float): Max gradient norm for clipping (default: 1.0)
        log_steps (int): Print loss every N steps (default: 100)
        model_name (str): Name for logging (default: 'lstm')
        compute_next_token_loss: Loss function from pico-llm.py
    
    Returns:
        MetricsTracker: Object containing all training metrics
    
    Example:
        >>> metrics = train_lstm_with_validation(
        ...     model, train_loader, val_loader, 
        ...     epochs=10, device=device, 
        ...     compute_next_token_loss=compute_next_token_loss
        ... )
        >>> metrics.plot_learning_curves('curves.png')
    """
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, lr=lr)
    
    # Metrics tracker
    metrics = MetricsTracker()
    
    print(f"\n{'='*70}")
    print(f"Training {model_name} with Validation")
    print(f"{'='*70}")
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Epochs: {epochs}, LR: {lr}, Grad Clip: {max_grad_norm}")
    print(f"{'='*70}\n")
    
    for epoch in range(1, epochs + 1):
        # ====== TRAINING PHASE ======
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch_tokens in enumerate(train_loader, start=1):
            batch_tokens = batch_tokens.to(device)
            
            # Forward pass
            logits = model(batch_tokens)
            loss = compute_next_token_loss(logits, batch_tokens)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            grad_norm = clip_gradients(model, max_norm=max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            
            # Track loss
            train_loss += loss.item()
            train_batches += 1
            
            # Periodic logging
            if batch_idx % log_steps == 0:
                avg_loss = train_loss / train_batches
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(train_loader)}, "
                      f"Loss: {avg_loss:.4f}, Grad: {grad_norm:.2f}")
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_batches
        
        # ====== VALIDATION PHASE ======
        avg_val_loss = validate_model(model, val_loader, device, compute_next_token_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update scheduler based on validation loss
        scheduler.step(avg_val_loss)
        
        # Track metrics
        metrics.add_epoch(epoch, avg_train_loss, avg_val_loss, current_lr)
        
        # ====== EPOCH SUMMARY ======
        gap = avg_val_loss - avg_train_loss
        print(f"\n{'='*70}")
        print(f"[{model_name}] Epoch {epoch}/{epochs} Summary:")
        print(f"  Train Loss:      {avg_train_loss:.4f}")
        print(f"  Val Loss:        {avg_val_loss:.4f}")
        print(f"  Overfitting Gap: {gap:.4f} {'⚠️ HIGH' if gap > 0.3 else '✓ OK'}")
        print(f"  Learning Rate:   {current_lr:.6f}")
        print(f"{'='*70}\n")
    
    # Print final summary
    final = metrics.get_final_metrics()
    print(f"\n{'='*70}")
    print(f"Training Complete - {model_name}")
    print(f"{'='*70}")
    print(f"Final Train Loss:     {final['final_train_loss']:.4f}")
    print(f"Final Val Loss:       {final['final_val_loss']:.4f}")
    print(f"Best Val Loss:        {final['best_val_loss']:.4f}")
    print(f"Avg Overfitting Gap:  {final['overfitting_gap']:.4f}")
    print(f"{'='*70}\n")
    
    return metrics


################################################################################
# 5. Our "stub" Transformer with KV-cache 
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        pass

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4):
        super().__init__()

        pass


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################
#Helpers 
def _get_token_embedding_weight(model):
    """
    Return an (vocab_size, d_model) embedding matrix if the model has one.
    """
    
    # Check if the model has an attribute named embedding
    if hasattr(model, "embedding") and isinstance(model.embedding, nn.Embedding):
        return model.embedding.weight

    # If the model doesn't directly store the embedding as model.embedding, 
    # we still scan through all submodules to find any nn.Embedding layer 
    # and return its weights.
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            return m.weight

    #If no embedding is found (like in the K-gram MLP, which doesn’t use embeddings)
    return None 

def _cosine_topk_rows(mat, row_vec, top_k):
    """
    mat: (N, D), row_vec: (D,)
    returns: list[(cos_sim, idx)] of length top_k sorted desc
    """
    #Normalize the each embedding vector in the matrix to unit length
    #1e-8 to avoid division by 0
    mat_norm = mat / (mat.norm(dim=1, keepdim=True) + 1e-8)

    #Normalize the target token’s embedding
    v = row_vec / (row_vec.norm() + 1e-8)

    #Compute cosine similarity between the normalized matrix rows and the normalized token vector.
    #Result: tensor of shape (N,), one similarity per token in the vocabulary.
    sims = (mat_norm @ v)  

    #Find the top-k most similar tokens.
    # values = cosine similarities
    # indices = token IDs (row indices) of the most similar embeddings.
    values, indices = torch.topk(sims, k=top_k)

    # Return a Python list of tuples (similarity, token_index).
    return [(values[i].item(), indices[i].item()) for i in range(values.numel())]

@torch.no_grad()
def monosemantic_analysis_for_token(token_id, model, device="cpu", top_n=5):
    """
    Return top-N nearest neighbors of token_id in embedding space.
    Output: list of tuples (similarity, neighbor_token_id).
    If the model has no embedding (e.g., K-gram MLP), return [].
    """
    #Get the embedding weight matrix
    W = _get_token_embedding_weight(model)
    #Failsafe: if the model has no embedding, return []
    if W is None:
        return [] 

    #Move the embedding matrix to the correct device
    W = W.to(device)

    # Sanity-check that the token_id is valid.
    token_id = int(token_id)
    if token_id < 0 or token_id >= W.size(0):
        return []

    # Extract that token’s embedding vector (shape (embedding_dim,))
    v = W[token_id] 

    # Compute cosine similarities between this token’s embedding and all others.
    #Ask for one extra (top_n + 1) so we can later remove the token itself.
    top_pairs = _cosine_topk_rows(W, v, top_k=min(top_n + 1, W.size(0)))

    # Build the result list by skipping the token itself (similarity = 1 with itself).
    #Keep adding until you collect exactly top_n neighbors.
    result = []
    for sim, idx in top_pairs:
        if idx == token_id:
            continue
        result.append((sim, idx))
        if len(result) >= top_n:
            break
    
    # Return a list of (similarity_score, neighbor_token_id) tuples.
    return result

def visualize_embeddings(model, sample_token_ids=None, max_points=800, title="Embedding PCA", savepath=None, device="cpu"):
    """
    Projects token embeddings to 2D with PCA (torch-only) and returns (coords_2d, used_token_ids).
    If matplotlib is available in your env, you can also scatter-plot and save.
    """
    W = _get_token_embedding_weight(model)
    if W is None:
        print("[viz] No embedding found on this model.")
        return None, None

    W = W.to(device)
    vocab_size = W.size(0)

    if sample_token_ids is None:
        if vocab_size > max_points:
            idx = torch.randperm(vocab_size, device=device)[:max_points]
        else:
            idx = torch.arange(vocab_size, device=device)
    else:
        idx = torch.tensor(sample_token_ids, dtype=torch.long, device=device)
        idx = idx[(idx >= 0) & (idx < vocab_size)]
        if idx.numel() == 0:
            print("[viz] Provided sample_token_ids are empty/invalid.")
            return None, None

    X = W[idx]  
    
    Xc = X - X.mean(dim=0, keepdim=True)

    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :2] * S[:2]  

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(Z[:, 0].cpu(), Z[:, 1].cpu(), s=4)
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        if savepath is not None:
            plt.savefig(savepath, bbox_inches="tight", dpi=150)
            print(f"[viz] Saved embedding PCA to {savepath}")
        plt.close()
    except Exception as e:
        print(f"[viz] Skipped plotting ({e})")

    return Z.detach().cpu(), idx.detach().cpu().tolist()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def evaluate_loss_and_ppl(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch_tokens in loader:
        batch_tokens = batch_tokens.to(device)
        logits = model(batch_tokens)
        loss = compute_next_token_loss(logits, batch_tokens)
        total_loss += loss.item()
        total_batches += 1
        if total_batches >= 50:   # cap for speed; tweak as needed
            break
    avg_loss = total_loss / max(total_batches, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl

def compare_models(models_dict, loader, device, title="Model Comparison", savepath=None):
    """
    Returns a summary dict and saves a bar chart of perplexities.
    """
    summary = {}
    names, ppls = [], []
    for name, model in models_dict.items():
        avg_loss, ppl = evaluate_loss_and_ppl(model, loader, device)
        n_params = count_parameters(model)
        summary[name] = {
            "avg_loss": avg_loss,
            "perplexity": ppl,
            "params": n_params,
        }
        names.append(name)
        ppls.append(ppl)

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.bar(names, ppls)
        plt.title(title)
        plt.ylabel("Perplexity (lower is better)")
        plt.xticks(rotation=15)
        if savepath is not None:
            plt.savefig(savepath, bbox_inches="tight", dpi=150)
            print(f"[compare] Saved chart to {savepath}")
        plt.close()
    except Exception as e:
        print(f"[compare] Skipped plotting ({e})")

    print("\n=== Model Comparison ===")
    for k, v in summary.items():
        print(f"{k:>16} | loss={v['avg_loss']:.4f} | ppl={v['perplexity']:.2f} | params={v['params']:,}")
    print("========================\n")

    return summary

def plot_attention_heatmap(attn_matrix, title="Attention", savepath=None):
    """
    attn_matrix: (T, T) or (H, T, T) torch tensor
    """
    try:
        import matplotlib.pyplot as plt
        A = attn_matrix.detach().float().cpu()
        if A.dim() == 3:
            # Show first head by default
            A = A[0]
        plt.figure()
        plt.imshow(A, aspect="auto")
        plt.title(title)
        plt.xlabel("Key positions")
        plt.ylabel("Query positions")
        if savepath:
            plt.savefig(savepath, bbox_inches="tight", dpi=150)
            print(f"[attn] Saved attention heatmap to {savepath}")
        plt.close()
    except Exception as e:
        print(f"[attn] Skipped plotting ({e})")


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    """
    Top-p (nucleus) sampling.

    Args:
        logits: (vocab_size,) unnormalized logits
        p: cumulative probability threshold

    Returns:
        token_id: int
    """
    # Convert logits to probabilities using softmax
    probs = F.softmax(logits, dim=-1)

    # Sort probabilities in descending order and get corresponding indices
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative sum of sorted probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find where cumulative probability exceeds threshold p
    cutoff_idx = torch.where(cumsum_probs >= p)[0]
    if len(cutoff_idx) > 0:
        # Take first index that exceeds p, add 1 to include it
        cutoff_idx = cutoff_idx[0].item() + 1
    else:
        # If no cutoff found, use all tokens
        cutoff_idx = len(sorted_probs)

    # Keep only top-p tokens (nucleus)
    top_probs = sorted_probs[:cutoff_idx]
    top_indices = sorted_indices[:cutoff_idx]

    # Renormalize probabilities to sum to 1
    top_probs = top_probs / top_probs.sum()

    # Sample from the nucleus according to renormalized probabilities
    sampled_idx = torch.multinomial(top_probs, num_samples=1).item()

    # Get the actual token ID from original vocabulary
    token_id = top_indices[sampled_idx].item()

    return token_id


def temperature_sampling(logits, temperature=1.0):
    """Temperature sampling - scale logits before softmax"""
    # Scale logits by temperature (higher temp = more random, lower temp = more deterministic)
    scaled_logits = logits / temperature

    # Convert scaled logits to probabilities
    probs = F.softmax(scaled_logits, dim=-1)

    # Sample one token from the probability distribution
    return torch.multinomial(probs, num_samples=1).item()


def top_k_sampling(logits, k=50):
    """Top-k sampling - sample from top k tokens"""
    # Get top k logits and their indices
    top_logits, top_indices = torch.topk(logits, k)

    # Convert top k logits to probabilities
    probs = F.softmax(top_logits, dim=-1)

    # Sample from top k according to their probabilities
    sampled_idx = torch.multinomial(probs, num_samples=1).item()

    # Return the actual token ID from original vocabulary
    return top_indices[sampled_idx].item()


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


################################################################################
# 8. Training
################################################################################

def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a"):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    next_sample_time = start_time
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=None,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=0.95,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # third generation => top-p=1.0 => full distribution random sampling
                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=1.0,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")


################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = 16
    num_epochs = 3
    learning_rate = 1e-3

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    ############################################################################
    # Models
    ############################################################################
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    transformer = TransformerModel(
    ).to(device)

    models = {
      # "kgram_mlp_seq": kgram_model,
        "lstm_seq": lstm_model,
      # "kvcache_transformer": kv_transformer,
    }


    monosemantic_info = {} if args.monosemantic_enabled else None


    ############################################################################
    # Train each model
    ############################################################################
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            monosemantic_info=monosemantic_info,
            prompt=args.prompt  # <--- Pass the user-specified prompt here
        )

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                monosemantic_info=monosemantic_info,
                do_monosemantic=(monosemantic_info is not None),
                top_p=None,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                monosemantic_info=monosemantic_info,
                do_monosemantic=(monosemantic_info is not None),
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                monosemantic_info=monosemantic_info,
                do_monosemantic=(monosemantic_info is not None),
                top_p=1.0,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    main()