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

#language model
word_embedding_dim=100
word_embedding_model="pretrain_word2vec/dim100/word2vec.bin"
lm_enc_dim=200
lm_dec_dim=600
lm_dec_layer_size=1
lm_attend_dim=25
lm_learning_rate=0.2

char_embedding_dim=150

pm_enc_dim=50
pm_dec_dim=200
pm_attend_dim=50
pm_learning_rate=0.001
repeat_loss_scale=1.0
cov_loss_scale=1.0
cov_loss_threshold=0.7
sigma=1.00

rm_dim=100
rm_num_context = 4 # number of context words for target word in rhyme model
rm_num_last_symbols = 10 # number of symbols from the end of the string to select target and context words
rm_delta=0.5
rm_learning_rate=0.001
