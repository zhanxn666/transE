# train_transe_pykeen.py
# this is a new branch test comment
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import torch
import os

# ==================== CONFIGURATION ====================
# Change these paths to match your data
train_path = "/home/e706/zhanxiangning/Tianchi/KnowledgeGraph/OpenBG500/OpenBG500_train.tsv"
valid_path = "/home/e706/zhanxiangning/Tianchi/KnowledgeGraph/OpenBG500/OpenBG500_dev.tsv"  # optional but recommended


# Output directory for results and trained model
result_dir = "./pykeen_transe_result"

# Model hyperparameters
embedding_dim = 100        # same as your old TransE
scoring_function = "l1"    # TransE uses L1 or L2 norm; l1 is common
num_epochs = 1         # increase to 500+ for better results
batch_size = 256
learning_rate = 0.01

# Use GPU if available
device = 'gpu' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ==================== LOAD DATA ====================
# PyKEEN expects tab-separated: head \t relation \t tail
# It will automatically create entity_to_id and relation_to_id mappings

triples_factory = TriplesFactory.from_path(
    path=train_path,
    delimiter='\t'
)

# Optional: validation and test splits
training = triples_factory
validation = None
testing = None

if os.path.exists(valid_path):
    validation = TriplesFactory.from_path(valid_path, delimiter='\t', 
                                          entity_to_id=training.entity_to_id,
                                          relation_to_id=training.relation_to_id)


# ==================== TRAIN TRANSE ====================
result = pipeline(
    model='TransE',
    model_kwargs=dict(
        embedding_dim=embedding_dim,
        scoring_fct_norm=1 if scoring_function == 'l1' else 2,  # 1 = L1, 2 = L2
    ),
    optimizer='adam',
    optimizer_kwargs=dict(lr=learning_rate),
    loss='marginranking',
    loss_kwargs=dict(margin=1.0),
    training=training,
    validation=None,
    testing=validation,
    training_kwargs=dict(
        num_epochs=num_epochs,
        batch_size=batch_size,
        sampler='default',  # or 'schlichtkrull' for large graphs
    ),
    evaluator='rankbased',
    evaluator_kwargs=dict(filtered=True),  # important: filtered metrics
    device=device,
    random_seed=42,
    use_tqdm=True,
)

# ==================== SAVE & PRINT RESULTS ====================
result.save_to_directory(result_dir)

print("\n=== Training Finished ===")
print(f"Best Hits@10 (filtered): {result.metric_results.get_metric('hits@10'):.4f}")
print(f"Best Mean Rank (filtered): {result.metric_results.get_metric('mr'):.2f}")
print(f"Best MRR: {result.metric_results.get_metric('mrr'):.4f}")

# Save entity and relation embeddings separately (like your old files)
model = result.model
entity_emb = model.entity_representations[0](indices=None).cpu().detach().numpy()
relation_emb = model.relation_representations[0](indices=None).cpu().detach().numpy()

# Save as .npy (binary, fast) or .txt (human-readable)
import numpy as np
np.save(os.path.join(result_dir, "entity_embeddings.npy"), entity_emb)
np.save(os.path.join(result_dir, "relation_embeddings.npy"), relation_emb)

# Optional: save as text like your old format
entity_ids = list(training.entity_to_id.keys())
relation_ids = list(training.relation_to_id.keys())

with open(os.path.join(result_dir, "entityVector.txt"), 'w', encoding='utf-8') as f:
    for name, vec in zip(entity_ids, entity_emb):
        vec_str = ', '.join(map(str, vec.tolist()))
        f.write(f"{name} [{vec_str}]\n")

with open(os.path.join(result_dir, "relationVector.txt"), 'w', encoding='utf-8') as f:
    for name, vec in zip(relation_ids, relation_emb):
        vec_str = ', '.join(map(str, vec.tolist()))
        f.write(f"{name} [{vec_str}]\n")

print(f"Embeddings saved to {result_dir}")