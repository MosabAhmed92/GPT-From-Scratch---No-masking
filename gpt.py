import torch 
import torch.nn as nn 
import math 
import torch.nn.functional as F
from torch.nn import Softmax

def compute_attention(Q, K, V):

    """
    Docstring for get_attention

    This function will be Executed after the Head split for that the shapes will be as follows : 
    
    :param Q: Query -> (batch, num_heads, seq_length, d_k)
    :param K: Key -> (Batch, num_heads, seq_length, d_k)
    :param V: Value -> (Batch, num_heads, seq_length, d_k)

    Outputs:

    weight, attention output 
    """

    # 1- compute the score -> Q @ K.T

    '''
    For the matrix multiplication Q @ K_t to work, you need:

    Q shape: (..., seq_length, d_k)
    K_t shape: (..., d_k, seq_length)
    '''

    K_t = torch.transpose(K, dim0 = 2, dim1 = 3) #(Batch, num_heads, d_k, seq_length)
    scores = Q @ K_t

    # 2 - scaling the score (divide the score by sqrt(d_model))
    d_k = Q.shape[-1]      
    scaled_score = scores/math.sqrt(d_k)

    # 3 - calculate the score weights ====> Softmax
    weights = F.softmax(scaled_score, dim = -1)

    # 4 - calculate the attention output
    attention_output = weights @ V

    return weights, attention_output


# Testing the Function 
# if __name__ == "__main__":

#     batch_size = 10
#     seq_length = 45
#     num_heads = 8
#     d_k = 128

#     Q = torch.randn(size = (batch_size, num_heads, seq_length, d_k))
#     K = torch.randn(size = (batch_size, num_heads, seq_length, d_k))
#     V = torch.randn(size = (batch_size, num_heads, seq_length, d_k))

#     weights, attention_output = compute_attention(Q, K, V)

#     print(f" Q shape: {Q.shape}")
#     print(f"weights shape: {weights.shape}")
#     print(f"attention output shape: {attention_output.shape}")



class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = self.d_model // self.num_heads

        # Find Linear weights for the Q, K, V 

        self.W_q = nn.Linear(in_features = d_model, out_features = d_model)
        self.W_k = nn.Linear(in_features = d_model, out_features = d_model)
        self.W_v = nn.Linear(in_features = d_model, out_features = d_model)
        self.W_o = nn.Linear(in_features = d_model, out_features = d_model)


    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # W_q(d_model, d_model) @ x (batch_size, seq_len, d_model)
        # `nn.Linear` applies to the **last dimension only**.
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        # Now Q, K, V (batch, seq_len, d_model) same shape of x but different values 

        # RESHAPING AND TRANSPOSING Q, K, V
        # (batch_size, seq_len, d_model) ===> (batch_size, num_heads, seq_length, d_k)

        # based on the number of heads of have
        d_k = self.d_model // self.num_heads

        Q = Q.reshape(batch_size, seq_len, self.num_heads, d_k)
        K = K.reshape(batch_size, seq_len, self.num_heads, d_k)
        V = V.reshape(batch_size, seq_len, self.num_heads, d_k)

        # Transpose seq_len, num_heads NOW : (batch_size, seq_len, num_heads, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)


        # Compute Attention

        _ , attention_output = compute_attention(Q, K, V)

        # attention_output ===> (Batch, num_heads, seq_length, d_k) 
        # we need to concatinate and get back to (batch_size, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(batch_size, seq_len, self.d_model)

        output = self.W_o(attention_output)

        return output
    
# if __name__ == '__main__':

#     batch_size = 10
#     seq_len = 45
#     d_model = 512
#     num_heads = 8

#     x = torch.randn(size = (batch_size, seq_len, d_model))

#     mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

#     output = mha(x)

#     print(f"Input shape : {x.shape}")
#     print(f"output shape : {output.shape}")


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.mha = MultiHeadAttention(self.d_model, self.num_heads)

        self.ffn1 = nn.Linear(in_features = self.d_model, out_features = self.d_ff)
        self.ffn2 = nn.Linear(in_features = self.d_ff, out_features = self.d_model)
        self.activation1 = nn.ReLU()

        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model)


    def forward(self, x):

        output = self.mha(x)

        output = x + output

        x1 = self.layer_norm1(output)
        output = self.activation1(self.ffn1(x1))

        ffn_output = self.ffn2(output)
        ffn_output = ffn_output + x1

        final_output = self.layer_norm2(ffn_output)

        return final_output
        

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,num_layers, d_ff, max_seq_len):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model)
        self.pos_embedding = nn.Embedding(num_embeddings = max_seq_len, embedding_dim = d_model)

        # stack of transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(d_model = self.d_model, num_heads = self.num_heads, d_ff = self.d_ff)
                                    for _ in range (num_layers)])
        
        self.final_layer_norm = nn.LayerNorm(self.d_model)

        # Output Projection

        self.final_layer = nn.Linear(in_features = self.d_model, out_features = self.vocab_size)


    def forward(self, x):
        # x shape : [batch, seq_len] - seq_len are the token IDs
        token_embeddings = self.embedding(x) # token_embedding (batch, seq_len, d_model)

        # Create pos_embeddings [0, 1, 2, ..., seq_len-1]
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device = x.device)
        pos_embeddings = self.pos_embedding(positions)    #(seq_len, d_model)

        final_embeddings = token_embeddings + pos_embeddings

        # pass through all Transformer Bloxks : 12 Blocks 
        output = final_embeddings
        for block in self.blocks:
            output = block(output) # output shape : (batch, seq_len, d_model)


        normalized_output = self.final_layer_norm(output) # output shape : (batch, seq_len, d_model)

        projection_output = self.final_layer(normalized_output) # projection output  : (batch, seq_len, vocab_size)


        return projection_output
    

if __name__ == '__main__':
    vocab_size = 50000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_len = 1024

    batch_size = 4
    seq_len = 100


    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    model  = GPT(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len)

    output = model(x)

    print(f"Input Shape is : {x.shape}")
    print(f"Output Shape is : {output.shape}")