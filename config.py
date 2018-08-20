seed=0
batch_size=32
keep_prob=0.7
max_grad_norm=5

poems_input_file='data/classic_poems.json'

num_epochs=100
output_dir='train'
save_model=True
save_model_steps=10000
restore_model_step=None
restore_model_latest=True

char_embedding_dim=150

rm_dim=100
rm_neg=5 #extra randomly sampled negative examples
rm_delta=0.5
rm_learning_rate=0.001
