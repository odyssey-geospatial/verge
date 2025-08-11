import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Tuple, Optional


class PermutationInvariantModel(nn.Module):
    """
    A permutation-invariant model that processes variable-length sets of vectors.
    Uses multi-head attention-like weighted aggregation followed by linear layers.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, 
                 num_attention_heads: int = 4, num_linear_layers: int = 2, dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of input vectors (C)
            hidden_dim: Hidden dimension for transformations
            embedding_dim: Final embedding dimension
            num_attention_heads: Number of attention heads (weighted sums)
            num_linear_layers: Number of linear layers after aggregation
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        
        # Transformation layer for input vectors
        self.transform = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention layer for computing weights (multiple columns)
        self.attention = nn.Linear(input_dim, num_attention_heads)
        
        # Linear layers after aggregation
        layers = []
        # Input dimension is now hidden_dim * num_attention_heads due to concatenation
        prev_dim = hidden_dim * num_attention_heads
        
        for i in range(num_linear_layers - 1):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final layer to embedding dimension
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.linear_layers = nn.Sequential(*layers)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, max_R, C)
            mask: Optional mask tensor of shape (batch_size, max_R) indicating valid positions
            
        Returns:
            embeddings: Tensor of shape (batch_size, embedding_dim)
        """
        batch_size, max_R, C = x.shape
        
        # Transform input vectors
        transformed = self.transform(x)  # (batch_size, max_R, hidden_dim)
        transformed = F.relu(transformed)
        transformed = self.dropout(transformed)
        
        # Compute multi-head attention weights
        attention_scores = self.attention(x)  # (batch_size, max_R, num_attention_heads)
        
        if mask is not None:
            # Apply mask (set invalid positions to large negative value)
            # Expand mask to match attention heads dimension
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.num_attention_heads)
            attention_scores = attention_scores.masked_fill(~mask_expanded, -1e9)
        
        # Apply softmax to get weights for each attention head
        weights = F.softmax(attention_scores, dim=1)  # (batch_size, max_R, num_attention_heads)
        
        # Compute weighted sum for each attention head
        # Expand transformed to match attention heads: (batch_size, max_R, hidden_dim, num_attention_heads)
        transformed_expanded = transformed.unsqueeze(-1).expand(-1, -1, -1, self.num_attention_heads)
        
        # Expand weights to match hidden dimension: (batch_size, max_R, hidden_dim, num_attention_heads)
        weights_expanded = weights.unsqueeze(2).expand(-1, -1, self.hidden_dim, -1)
        
        # Weighted sum for each head: (batch_size, hidden_dim, num_attention_heads)
        weighted_sums = torch.sum(weights_expanded * transformed_expanded, dim=1)
        
        # Concatenate all attention heads: (batch_size, hidden_dim * num_attention_heads)
        aggregated = weighted_sums.view(batch_size, -1)
        
        # Pass through linear layers
        embeddings = self.linear_layers(aggregated)  # (batch_size, embedding_dim)
        
        # L2 normalize embeddings for contrastive learning
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class TripletContrastiveLoss(nn.Module):
    """
    Contrastive loss for explicit triplet sampling.
    """
    
    def __init__(self, margin: float = 0.5, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, anchor_emb: torch.Tensor, pos_emb: torch.Tensor, 
                neg_emb: torch.Tensor, neg_batch_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet contrastive loss.
        
        Args:
            anchor_emb: Anchor embeddings (batch_size, embedding_dim)
            pos_emb: Positive embeddings (batch_size, embedding_dim)  
            neg_emb: Negative embeddings (total_negatives, embedding_dim)
            neg_batch_indices: Which batch item each negative belongs to (total_negatives,)
            
        Returns:
            loss: Contrastive loss value
        """
        batch_size = anchor_emb.shape[0]
        
        # Compute positive similarities (anchor-positive pairs)
        pos_sim = F.cosine_similarity(anchor_emb, pos_emb, dim=1)  # (batch_size,)
        
        # Compute negative similarities (anchor-negative pairs)
        total_loss = torch.tensor(0.0, device=anchor_emb.device, requires_grad=True)
        
        for i in range(batch_size):
            # Get negatives for this batch item
            neg_mask = neg_batch_indices == i
            if neg_mask.sum() == 0:
                continue
                
            batch_negatives = neg_emb[neg_mask]  # (num_negs_for_this_item, embedding_dim)
            anchor_i = anchor_emb[i:i+1]  # (1, embedding_dim)
            
            # Compute similarities with all negatives for this anchor
            neg_sims = F.cosine_similarity(
                anchor_i.expand_as(batch_negatives), batch_negatives, dim=1
            )  # (num_negs_for_this_item,)
            
            # Triplet loss: max(0, margin + neg_sim - pos_sim)
            pos_sim_i = pos_sim[i]
            triplet_losses = torch.clamp(self.margin + neg_sims - pos_sim_i, min=0.0)
            
            # Average over negatives for this anchor
            if len(triplet_losses) > 0:
                total_loss = total_loss + triplet_losses.mean()
        
        return total_loss / batch_size


class ContrastivePairDataset(Dataset):
    """
    Dataset for explicit contrastive pair sampling.
    Each sample returns an anchor, positive, and negative triplet.
    """
    
    def __init__(self, sequences: List[np.ndarray], similarity_features: np.ndarray, 
                 similarity_threshold: float = 0.5, num_negatives: int = 1):
        """
        Args:
            sequences: List of numpy arrays, each of shape (R_i, C)
            similarity_features: Array of shape (num_instances, feature_dim)
            similarity_threshold: Threshold for determining positive pairs
            num_negatives: Number of negative samples per anchor
        """
        self.sequences = sequences
        self.similarity_features = similarity_features
        self.similarity_threshold = similarity_threshold
        self.num_negatives = num_negatives
        
        assert len(sequences) == len(similarity_features)
        
        # Pre-compute similarity matrix for efficient sampling
        self._compute_similarity_matrix()
        self._build_positive_negative_maps()
    
    def _compute_similarity_matrix(self):
        """Compute pairwise similarity matrix from similarity features."""
        # Normalize features
        features_norm = self.similarity_features / np.linalg.norm(
            self.similarity_features, axis=1, keepdims=True)
        
        # Compute cosine similarity matrix
        self.similarity_matrix = np.dot(features_norm, features_norm.T)
    
    def _build_positive_negative_maps(self):
        """Build maps of positive and negative indices for each instance."""
        n = len(self.sequences)
        self.positive_map = {}
        self.negative_map = {}
        
        for i in range(n):
            # Find positive samples (excluding self)
            positive_mask = (self.similarity_matrix[i] > self.similarity_threshold) & (np.arange(n) != i)
            self.positive_map[i] = np.where(positive_mask)[0]
            
            # Find negative samples
            negative_mask = (self.similarity_matrix[i] <= self.similarity_threshold) & (np.arange(n) != i)
            self.negative_map[i] = np.where(negative_mask)[0]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, anchor_idx):
        """
        Returns anchor, positive, and negative samples.
        """
        anchor_seq = self.sequences[anchor_idx]
        anchor_sim = self.similarity_features[anchor_idx]
        
        # Sample a positive (if any exist)
        if len(self.positive_map[anchor_idx]) > 0:
            pos_idx = np.random.choice(self.positive_map[anchor_idx])
            pos_seq = self.sequences[pos_idx]
            pos_sim = self.similarity_features[pos_idx]
        else:
            # If no positives, use the anchor itself (self-positive)
            pos_idx = anchor_idx
            pos_seq = anchor_seq.copy()
            pos_sim = anchor_sim.copy()
        
        # Sample negatives
        if len(self.negative_map[anchor_idx]) >= self.num_negatives:
            neg_indices = np.random.choice(
                self.negative_map[anchor_idx], size=self.num_negatives, replace=False)
        else:
            # If not enough negatives, sample with replacement
            neg_indices = np.random.choice(
                self.negative_map[anchor_idx], size=self.num_negatives, replace=True)
        
        neg_seqs = [self.sequences[idx] for idx in neg_indices]
        neg_sims = [self.similarity_features[idx] for idx in neg_indices]
        
        return {
            'anchor': (anchor_seq, anchor_sim),
            'positive': (pos_seq, pos_sim),
            'negatives': [(seq, sim) for seq, sim in zip(neg_seqs, neg_sims)]
        }


def triplet_collate_fn(batch):
    """
    Collate function for triplet data (anchor, positive, negatives).
    """
    anchors, positives, negatives_list = [], [], []
    
    for item in batch:
        anchor_seq, anchor_sim = item['anchor']
        pos_seq, pos_sim = item['positive']
        neg_items = item['negatives']
        
        anchors.append((anchor_seq, anchor_sim))
        positives.append((pos_seq, pos_sim))
        negatives_list.append(neg_items)
    
    # Process anchors
    anchor_seqs, anchor_sims = zip(*anchors)
    anchor_padded, anchor_masks = _pad_sequences(anchor_seqs)
    anchor_sims = np.array(anchor_sims)
    
    # Process positives
    pos_seqs, pos_sims = zip(*positives)
    pos_padded, pos_masks = _pad_sequences(pos_seqs)
    pos_sims = np.array(pos_sims)
    
    # Process negatives (flatten all negatives from all batch items)
    all_neg_seqs, all_neg_sims = [], []
    neg_batch_indices = []  # Track which batch item each negative belongs to
    
    for batch_idx, neg_items in enumerate(negatives_list):
        for neg_seq, neg_sim in neg_items:
            all_neg_seqs.append(neg_seq)
            all_neg_sims.append(neg_sim)
            neg_batch_indices.append(batch_idx)
    
    neg_padded, neg_masks = _pad_sequences(all_neg_seqs)
    all_neg_sims = np.array(all_neg_sims)
    neg_batch_indices = np.array(neg_batch_indices)
    
    return {
        'anchor': (torch.FloatTensor(anchor_padded), torch.BoolTensor(anchor_masks), 
                   torch.FloatTensor(anchor_sims)),
        'positive': (torch.FloatTensor(pos_padded), torch.BoolTensor(pos_masks), 
                     torch.FloatTensor(pos_sims)),
        'negatives': (torch.FloatTensor(neg_padded), torch.BoolTensor(neg_masks), 
                      torch.FloatTensor(all_neg_sims), torch.LongTensor(neg_batch_indices))
    }


def _pad_sequences(sequences):
    """Helper function to pad sequences and create masks."""
    max_len = max(seq.shape[0] for seq in sequences)
    batch_size = len(sequences)
    feature_dim = sequences[0].shape[1]
    
    padded_sequences = np.zeros((batch_size, max_len, feature_dim))
    masks = np.zeros((batch_size, max_len), dtype=bool)
    
    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        padded_sequences[i, :seq_len] = seq
        masks[i, :seq_len] = True
    
    return padded_sequences, masks


def train_model_with_triplets(model, train_loader, num_epochs=100, learning_rate=1e-3, device='cpu'):
    """
    Training loop for the permutation-invariant model with explicit triplet sampling.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = TripletContrastiveLoss(margin=0.5, temperature=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Unpack batch data
            anchor_seqs, anchor_masks, anchor_sims = batch_data['anchor']
            pos_seqs, pos_masks, pos_sims = batch_data['positive']
            neg_seqs, neg_masks, neg_sims, neg_batch_indices = batch_data['negatives']
            
            # Move to device
            anchor_seqs, anchor_masks = anchor_seqs.to(device), anchor_masks.to(device)
            pos_seqs, pos_masks = pos_seqs.to(device), pos_masks.to(device)
            neg_seqs, neg_masks = neg_seqs.to(device), neg_masks.to(device)
            neg_batch_indices = neg_batch_indices.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            anchor_emb = model(anchor_seqs, anchor_masks)
            pos_emb = model(pos_seqs, pos_masks)
            neg_emb = model(neg_seqs, neg_masks)
            
            # Compute loss
            loss = criterion(anchor_emb, pos_emb, neg_emb, neg_batch_indices)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')


# Keep the old training function for backward compatibility
def train_model(model, train_loader, num_epochs=100, learning_rate=1e-3, device='cpu'):
    """
    Original training loop (for backward compatibility).
    """
    return train_model_with_triplets(model, train_loader, num_epochs, learning_rate, device)


# Improved synthetic data generation with meaningful patterns
def generate_structured_synthetic_data(num_instances=1000, min_R=5, max_R=20, C=32, 
                                     similarity_dim=16, num_clusters=5, noise_level=0.1):
    """
    Generate synthetic data with meaningful patterns that can be learned.
    
    Args:
        num_instances: Number of instances to generate
        min_R, max_R: Range for sequence lengths
        C: Fixed dimension of each vector in sequences
        similarity_dim: Dimension of similarity features
        num_clusters: Number of distinct clusters/patterns
        noise_level: Amount of noise to add (0 = no noise, 1 = high noise)
    
    Returns:
        sequences: List of variable-length sequences
        similarity_features: Array of similarity features that relate to sequence patterns
    """
    sequences = []
    similarity_features = []
    
    # Create cluster centers for both input patterns and similarity features
    input_cluster_centers = np.random.randn(num_clusters, C) * 2  # Spread out centers
    similarity_cluster_centers = np.random.randn(num_clusters, similarity_dim) * 2
    
    # Additional pattern parameters for each cluster
    cluster_patterns = []
    for i in range(num_clusters):
        pattern = {
            'center': input_cluster_centers[i],
            'sim_center': similarity_cluster_centers[i],
            'frequency_pattern': np.random.choice([0.3, 0.5, 0.8]),  # How often the pattern appears
            'scale_factor': np.random.uniform(0.5, 2.0),  # Scaling of the pattern
            'trend_direction': np.random.choice([-1, 0, 1]) * 0.1,  # Linear trend in sequences
        }
        cluster_patterns.append(pattern)
    
    instance_clusters = []  # Track which cluster each instance belongs to
    
    for i in range(num_instances):
        # Assign instance to a cluster
        cluster_id = np.random.randint(0, num_clusters)
        instance_clusters.append(cluster_id)
        pattern = cluster_patterns[cluster_id]
        
        # Generate sequence length
        R = np.random.randint(min_R, max_R + 1)
        
        # Generate sequence with meaningful patterns
        sequence = generate_patterned_sequence(
            R, C, pattern, noise_level
        )
        sequences.append(sequence.astype(np.float32))
        
        # Generate similarity features that relate to the sequence pattern
        sim_feature = generate_similarity_feature(
            sequence, pattern, similarity_dim, noise_level
        )
        similarity_features.append(sim_feature)
    
    similarity_features = np.array(similarity_features, dtype=np.float32)
    
    print(f"Generated {num_instances} instances across {num_clusters} clusters")
    print(f"Cluster distribution: {np.bincount(instance_clusters)}")
    
    return sequences, similarity_features, instance_clusters


def generate_patterned_sequence(R, C, pattern, noise_level):
    """Generate a sequence with meaningful patterns."""
    sequence = np.zeros((R, C))
    
    center = pattern['center']
    freq = pattern['frequency_pattern']
    scale = pattern['scale_factor']
    trend = pattern['trend_direction']
    
    for r in range(R):
        # Base pattern from cluster center
        base_vector = center * scale
        
        # Add frequency-based variation
        freq_component = np.sin(2 * np.pi * freq * r / R) * 0.5
        base_vector = base_vector + freq_component
        
        # Add linear trend
        trend_component = trend * (r / R) * np.ones(C)
        base_vector = base_vector + trend_component
        
        # Add some position-dependent variation
        position_noise = np.random.randn(C) * 0.2
        
        # Add controlled noise
        noise = np.random.randn(C) * noise_level
        
        sequence[r] = base_vector + position_noise + noise
    
    return sequence


def generate_similarity_feature(sequence, pattern, similarity_dim, noise_level):
    """Generate similarity features that relate to sequence characteristics."""
    
    # Extract meaningful statistics from the sequence
    seq_mean = np.mean(sequence, axis=0)  # Overall mean
    seq_std = np.std(sequence, axis=0)   # Variability
    seq_trend = sequence[-1] - sequence[0] if len(sequence) > 1 else np.zeros_like(sequence[0])  # Trend
    
    # Dimensionality reduction: take first few components
    reduced_features = []
    feature_components = [seq_mean, seq_std, seq_trend]
    
    for component in feature_components:
        # Take first few dimensions and pad if needed
        comp_size = min(len(component), similarity_dim // 3)
        reduced_features.extend(component[:comp_size])
    
    # Pad to desired dimension
    while len(reduced_features) < similarity_dim:
        reduced_features.append(0.0)
    
    # Truncate if too long
    reduced_features = reduced_features[:similarity_dim]
    
    # Add cluster-specific signal
    cluster_signal = pattern['sim_center']
    if len(cluster_signal) >= similarity_dim:
        cluster_signal = cluster_signal[:similarity_dim]
    else:
        # Pad cluster signal
        cluster_signal = np.concatenate([
            cluster_signal, 
            np.zeros(similarity_dim - len(cluster_signal))
        ])
    
    # Combine sequence-derived features with cluster signal
    similarity_feature = 0.7 * np.array(reduced_features) + 0.3 * cluster_signal
    
    # Add noise
    noise = np.random.randn(similarity_dim) * noise_level * 0.5
    similarity_feature = similarity_feature + noise
    
    return similarity_feature.astype(np.float32)


def generate_evaluation_data(num_instances=200, **kwargs):
    """Generate a smaller dataset for evaluation with the same patterns."""
    return generate_structured_synthetic_data(num_instances=num_instances, **kwargs)


# Example usage and data generation (updated)
def generate_sample_data(num_instances=1000, min_R=5, max_R=20, C=32, similarity_dim=16):
    """
    Legacy function - now calls the improved structured data generator.
    """
    print("Using improved structured synthetic data generation...")
    sequences, similarity_features, clusters = generate_structured_synthetic_data(
        num_instances=num_instances,
        min_R=min_R, 
        max_R=max_R, 
        C=C, 
        similarity_dim=similarity_dim,
        num_clusters=max(5, num_instances // 200),  # Scale clusters with data size
        noise_level=0.2
    )
    return sequences, similarity_features


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate sample data
    sequences, similarity_features = generate_sample_data(
        num_instances=1000, min_R=5, max_R=20, C=32, similarity_dim=16
    )
    
    # Create dataset and dataloader with explicit triplet sampling
    dataset = ContrastivePairDataset(
        sequences, 
        similarity_features, 
        similarity_threshold=0.5,  # Adjust based on your similarity features
        num_negatives=2  # Number of negatives per anchor
    )
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=triplet_collate_fn)
    
    # Initialize model
    model = PermutationInvariantModel(
        input_dim=32,
        hidden_dim=128,
        embedding_dim=64,
        num_attention_heads=4,  # Now using 4 attention heads
        num_linear_layers=3,
        dropout=0.1
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train the model
    train_model(model, train_loader, num_epochs=50, learning_rate=1e-3, device=device)
    
    # Example inference
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        # Unpack the dictionary structure from triplet data loader
        anchor_seqs, anchor_masks, anchor_sims = sample_batch['anchor']
        anchor_seqs = anchor_seqs.to(device)
        anchor_masks = anchor_masks.to(device)
        
        # Generate embeddings for anchor samples
        embeddings = model(anchor_seqs, anchor_masks)
        print(f"Generated embeddings shape: {embeddings.shape}")
        print(f"Sample embedding norm: {torch.norm(embeddings[0]).item():.4f}")
        
        # You can also get embeddings for positives and negatives
        pos_seqs, pos_masks, pos_sims = sample_batch['positive']
        pos_seqs, pos_masks = pos_seqs.to(device), pos_masks.to(device)
        pos_embeddings = model(pos_seqs, pos_masks)
        print(f"Positive embeddings shape: {pos_embeddings.shape}")
        
        # Check similarity between anchors and positives
        similarities = F.cosine_similarity(embeddings, pos_embeddings, dim=1)
        print(f"Anchor-Positive similarities: {similarities.mean().item():.4f} ± {similarities.std().item():.4f}")