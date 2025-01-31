from Transformer_Latent import create_cnn_transformer_model
from Transformer_Compile import Trans_Compile
import pandas as pd

filters = [
    [32, 64, 128],  # Filter configuration 1
    [64, 128, 256],  # Filter configuration 2
    [64, 256, 512],  # Filter configuration 3
]

dropout_rates = [0.3, 0.5]  # List of dropout rates to try
attention_heads = [4, 8, 12]  # List of attention head configurations
dope = [128, 256, 512]  # List of feed-forward layer dimensions
LatentDim = [256, 512]  # List of latent space dimensions

# Perform manual hyperparameter tuning
best_test_acc, best_Filter, best_do_rate, best_att_head, best_latent_dim, best_dense_dim, results, best_model, history = Trans_Compile(
    filters=filters,
    dropout_rates=dropout_rates,
    attention_heads=attention_heads,
    dope=dope,
    LatentDim=LatentDim
)

best_results_dict = {'testAccuracy':best_test_acc, 'Filter': best_Filter, 'dropOutRate':best_do_rate, 'attentionHead':best_att_head, 'latentDimensions':best_latent_dim, 'denseDimensions':best_dense_dim}

results_df = pd.DataFrame(best_results_dict)