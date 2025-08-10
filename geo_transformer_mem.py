import numpy as np
import copy
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import pickle

# This class wraps a list of input tile data as a pytorch dataset.
# The "getitem" method here parses apart the true labels and the encodings,
# and applies random masking to the encoding.

# This "mem" version keeps the whole dataset in memory.

class VergeDataset(torch.utils.data.Dataset):

    def __init__(self, aoi_fnames, n_classes, mask_fraction, class_prob):
        self.aoi_fnames = aoi_fnames
        self.n_classes = n_classes
        self.mask_fraction = mask_fraction
        self.class_prob = class_prob
        
        # Read all instances.
        self.instances = []
        for aoi_fname in aoi_fnames:
            with open(aoi_fname, 'rb') as source:
                loaded = pickle.load(source)
            print('loaded %d instances from %s' % (len(loaded), aoi_fname))
            self.instances += list(loaded)
            print('%d instances total' % len(self.instances))

    
    def __len__(self):
        return len(self.instances)

    
    def __getitem__(self, idx):

        # Get the features and labels from a file.
        loaded = self.instances[idx]
        features = loaded['features']
        true_labels = loaded['labels']

        encodings = features[:, self.n_classes:]
        true_labels_onehot = features[:, :self.n_classes]
        n_entities = features.shape[0]

        # Sample entities to mask. This weights the sampling by the relative
        # frequency of different classes in the dataset -- i.e. it addresses
        # class imbalance.
        weights = []
        for label in true_labels:
            prob = self.class_prob[label]
            weights.append(1.0 / (prob + 0.00001))
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        sample_size = int(np.ceil(self.mask_fraction * n_entities))
        mask_indices = np.random.choice(n_entities, size=sample_size, replace=True, p=weights)

        # In the feature array, labels are one-hot vectors that get concatenated
        # with the geometric encodings. To "mask" those labels, we replace the
        # one-hot vector with a zero-hot vector.
        mask_vector = np.zeros(self.n_classes)
        masked_labels_onehot = copy.copy(true_labels_onehot)
        for i in mask_indices:
            masked_labels_onehot[i] = mask_vector
            # print('replaced one-hot vectdor for row %d' % i)

        # Re-concatenate the masked labels with the geometric encodings.
        masked_labels_onehot_tensor = torch.tensor(masked_labels_onehot, dtype=torch.float32)
        encodings_tensor = torch.tensor(encodings, dtype=torch.float32)
        masked_features = torch.cat(
            [masked_labels_onehot_tensor, encodings_tensor], dim=1
        )

        # During model training below, we will be using the "CrossEntropyLoss" function,
        # which has a built-in capability to ignore elements that we don't care about,
        # which in this case is any element that is NOT masked. To get it to work,
        # we need to pack an "ignore" token into any label slot that is not masked.
        # Pytorch's standard value for that token is -100. Or more specifically
        # we start with all "ignore" tokens and just replace the ones that we do
        # care about with the appropriate value.
        labels = torch.full(true_labels.shape, -100, dtype=torch.long)
        for i in mask_indices:
            labels[i] = true_labels[i]
            # print('set true label for element %d to %d' % (i, true_labels[i]))00

        # Shuffle the features and labels.
        perm = torch.randperm(masked_features.shape[0])
        masked_features = masked_features[perm]
        labels = labels[perm]

        return (masked_features, labels)


# Define the function that puts together a batch. The main thing we are handling here
# is padding. We make all arrays have a size equal to the largest one in the batch,
# with excess space filled with padding tokens.
def verge_collate_fn(batch):

    features, labels = zip(*batch)
    max_len = max(x.shape[0] for x in features)
    batch_size = len(features)
    feature_dim = features[0].shape[1]

    padded_features = torch.zeros(batch_size, max_len, feature_dim)
    padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # -100 is the "ignore" value
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i in range(batch_size):
        n = features[i].shape[0]
        padded_features[i, :n] = features[i]
        padded_labels[i, :n] = labels[i]
        attention_mask[i, :n] = 1

    return padded_features, padded_labels, attention_mask




class GeospatialTransformer(nn.Module):

    def __init__(self, feature_dim, model_dim, num_heads, num_layers, num_classes, dropout):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(model_dim, num_classes)


    def forward(self, x, attention_mask):
        """
        x: Tensor of shape [batch_size, n_entities, encoding_dim]
        attention_mask: Tensor of shape [batch_size, n_entities], with 1 for valid, 0 for padding
        """
        # print('input', x.shape)

        x = self.input_proj(x)
        # print('projected', x.shape)

        # Transformer expects padding mask: True for PAD tokens
        pad_mask = (attention_mask == 0)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        # print('transformed', x.shape)

        # x = torch.flatten(x, start_dim=1)
        # print('flattened', x.shape)

        logits = self.output_head(x)
        # print('logits', logits.shape)

        return logits


    def embed(self, x, attention_mask):
        """
        Returns an embedding for the input features
        """
        x = self.input_proj(x)
        pad_mask = (attention_mask == 0)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return x






