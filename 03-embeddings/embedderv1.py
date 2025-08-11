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


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training embeddings.
    Uses cosine similarity and InfoNCE-style loss.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, similarity_features: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Model embeddings of shape (batch_size, embedding_dim)
            similarity_features: Features for computing similarity of shape (batch_size, feature_dim)
            
        Returns:
            loss: Contrastive loss value
        """
        batch_size = embeddings.shape[0]
        
        # Compute similarity matrix from similarity features
        # Using cosine similarity
        similarity_features_norm = F.normalize(similarity_features, p=2, dim=1)
        target_similarity = torch.mm(similarity_features_norm, similarity_features_norm.t())
        
        # Compute embedding similarity matrix
        embedding_similarity = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Create positive pairs mask (similar instances should be close)
        # We consider pairs with similarity > threshold as positive
        threshold = 0.5  # Adjust based on your similarity features
        positive_mask = (target_similarity > threshold) & ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        
        # InfoNCE-style loss
        # For each instance, positive pairs should have higher similarity than negative pairs
        loss = 0.0
        num_positives = 0
        
        for i in range(batch_size):
            if positive_mask[i].sum() > 0:  # If there are positive pairs for this instance
                positive_scores = embedding_similarity[i][positive_mask[i]]
                negative_scores = embedding_similarity[i][~positive_mask[i] & ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)[i]]
                
                # Compute InfoNCE loss for this instance
                if len(negative_scores) > 0:
                    logits = torch.cat([positive_scores, negative_scores])
                    labels = torch.zeros(len(logits), device=embeddings.device)
                    labels[:len(positive_scores)] = 1.0
                    
                    # Binary cross-entropy with logits
                    instance_loss = F.binary_cross_entropy_with_logits(logits, labels)
                    loss += instance_loss
                    num_positives += 1
        
        return loss / max(num_positives, 1)


class VariableLengthDataset(Dataset):
    """
    Dataset for handling variable-length sequences with similarity features.
    """
    
    def __init__(self, sequences: List[np.ndarray], similarity_features: np.ndarray):
        """
        Args:
            sequences: List of numpy arrays, each of shape (R_i, C)
            similarity_features: Array of shape (num_instances, feature_dim)
        """
        self.sequences = sequences
        self.similarity_features = similarity_features
        assert len(sequences) == len(similarity_features)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.similarity_features[idx]


def collate_fn(batch):
    """
    Collate function for DataLoader to handle variable-length sequences.
    """
    sequences, similarity_features = zip(*batch)
    
    # Find maximum sequence length in batch
    max_len = max(seq.shape[0] for seq in sequences)
    batch_size = len(sequences)
    feature_dim = sequences[0].shape[1]
    
    # Pad sequences and create masks
    padded_sequences = np.zeros((batch_size, max_len, feature_dim))
    masks = np.zeros((batch_size, max_len), dtype=bool)
    
    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        padded_sequences[i, :seq_len] = seq
        masks[i, :seq_len] = True
    
    return (torch.FloatTensor(padded_sequences), 
            torch.BoolTensor(masks),
            torch.FloatTensor(np.array(similarity_features)))


def train_model(model, train_loader, num_epochs=100, learning_rate=1e-3, device='cpu'):
    """
    Training loop for the permutation-invariant model.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = ContrastiveLoss(temperature=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (sequences, masks, similarity_features) in enumerate(train_loader):
            sequences = sequences.to(device)
            masks = masks.to(device)
            similarity_features = similarity_features.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = model(sequences, masks)
            
            # Compute loss
            loss = criterion(embeddings, similarity_features)
            
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


# Example usage and data generation
def generate_sample_data(num_instances=1000, min_R=5, max_R=20, C=32, similarity_dim=16):
    """
    Generate sample data for testing.
    """
    sequences = []
    similarity_features = []
    
    for i in range(num_instances):
        R = np.random.randint(min_R, max_R + 1)
        sequence = np.random.randn(R, C).astype(np.float32)
        sequences.append(sequence)
        
        # Generate similarity features
        sim_feature = np.random.randn(similarity_dim).astype(np.float32)
        similarity_features.append(sim_feature)
    
    return sequences, np.array(similarity_features)


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate sample data
    sequences, similarity_features = generate_sample_data(
        num_instances=1000, min_R=5, max_R=20, C=32, similarity_dim=16
    )
    
    # Create dataset and dataloader
    dataset = VariableLengthDataset(sequences, similarity_features)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
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
        sequences, masks, similarity_features = sample_batch
        sequences = sequences.to(device)
        masks = masks.to(device)
        
        embeddings = model(sequences, masks)
        print(f"Generated embeddings shape: {embeddings.shape}")
        print(f"Sample embedding norm: {torch.norm(embeddings[0]).item():.4f}")