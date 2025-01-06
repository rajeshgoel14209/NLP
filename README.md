Inputs in LLM Fine-tuning
Text Data:

Fine-tuning typically requires a dataset with input-output pairs, such as:
Plain text (for language modeling).
Question-answer pairs (for QA tasks).
Prompts and responses (for instruction-following tasks).
Context and completion pairs (for generative tasks).
Format:
Unsupervised Fine-tuning: Continuous text (e.g., books, articles).
Supervised Fine-tuning: Structured pairs (e.g., prompts and labels).
Tokenized Inputs:

Raw text is tokenized into subword units (tokens) using the tokenizer associated with the LLM (e.g., Byte Pair Encoding, SentencePiece).
For sequence-to-sequence models, both input and output texts are tokenized.
Attention Masks:

Specify which tokens the model should attend to (important for handling padding in batches).
Labels:

In supervised tasks, labels are the ground truth outputs. For generative tasks, these are often tokenized versions of the desired outputs.
Special Tokens:

Tokens for task-specific signals, such as <BOS> (beginning of sentence), <EOS> (end of sentence), or separator tokens (<SEP>).
Loss Functions in LLM Fine-tuning
The choice of loss function depends on the task:

1. Causal Language Modeling (CLM)
Task: Predict the next token given previous tokens (autoregressive modeling).
Loss Function:
Cross-Entropy Loss over the predicted token probabilities and the ground truth token.
2. Masked Language Modeling (MLM)
Task: Predict masked tokens in a sequence (e.g., BERT-style models).
Loss Function:
Cross-Entropy Loss computed only over the masked tokens.
3. Sequence-to-Sequence (Seq2Seq) Learning
Task: Translate an input sequence into an output sequence (e.g., summarization, translation).
Loss Function:
Cross-Entropy Loss applied to the decoder's outputs and the ground truth tokens.
4. Token Classification
Task: Assign a label to each token in the input (e.g., named entity recognition, part-of-speech tagging).
Loss Function:
Cross-Entropy Loss computed at the token level.
5. Sequence Classification
Task: Assign a single label to an entire sequence (e.g., sentiment analysis, text classification).
Loss Function:
Cross-Entropy Loss computed at the sequence level.
6. Contrastive Loss
Task: Learn embeddings where similar inputs are closer in vector space (e.g., sentence embeddings, retrieval tasks).
Loss Function:
Contrastive Loss, such as Triplet Loss or NT-Xent Loss.
7. Reinforcement Learning with Human Feedback (RLHF)
Task: Fine-tune the model's outputs to align with human preferences.
Loss Function:
A combination of reward models and policy gradients (e.g., PPO) to optimize the model's policy.
