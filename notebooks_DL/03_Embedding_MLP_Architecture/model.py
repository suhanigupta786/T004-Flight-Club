import torch
import torch.nn as nn


class FlightDelayMLP(nn.Module):
    """MLP for flight-delay prediction using categorical embeddings.

    Args:
        numerical_input_dim: Number of numerical input features.
        category_sizes: Dictionary mapping each categorical feature name to
            its embedding-table size (number of category indices).
        embedding_dims: Dictionary mapping each categorical feature name to
            its embedding dimension.
        hidden_dims: Sizes of the fully connected hidden layers.
        dropout_rates: Dropout rates applied after the corresponding hidden
            layers. The defaults match Notebook 03.

    Inputs to forward:
        x_num: Float tensor with shape (batch_size, numerical_input_dim).
        x_cat: Long tensor with shape (batch_size, number_of_categorical_features).
              Its columns must follow the same order as category_sizes keys.

    Returns:
        Raw prediction logits with shape (batch_size,). No sigmoid is applied.
    """

    def __init__(
        self,
        numerical_input_dim,
        category_sizes,
        embedding_dims,
        hidden_dims=(128, 64, 32),
        dropout_rates=(0.3, 0.2)
    ):
        super().__init__()

        self.num_categorical_features = len(category_sizes)

        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=category_sizes[feature],
                embedding_dim=embedding_dims[feature]
            )
            for feature in category_sizes
        ])

        total_embedding_dim = sum(embedding_dims.values())

        combined_input_dim = (
            numerical_input_dim + total_embedding_dim
        )

        layers = []
        previous_dim = combined_input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())

            if i < len(dropout_rates):
                layers.append(nn.Dropout(dropout_rates[i]))

            previous_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(previous_dim, 1)

    def forward(self, x_num, x_cat):
        embedded_features = []

        for i, embedding_layer in enumerate(self.embeddings):
            embedded = embedding_layer(x_cat[:, i])
            embedded_features.append(embedded)

        categorical_output = torch.cat(embedded_features, dim=1)
        combined_input = torch.cat([x_num, categorical_output], dim=1)
        hidden_output = self.mlp(combined_input)
        logits = self.output_layer(hidden_output)

        return logits.squeeze(1)
