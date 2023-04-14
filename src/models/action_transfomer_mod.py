from torch import nn
import torch


class ActionTransformer(nn.Module):
    def __init__(self,nhead,num_layers,d_keypoints=17*2,nb_actions=4) -> None:
        super().__init__()

        self.d_keypoints = d_keypoints

        self.nhead = nhead
        self.d_model = 64*self.nhead
        self.num_layers = num_layers
        self.Dmlp = 4*self.d_model
        self.nb_actions = nb_actions

        self.embedding_layer = nn.Linear(self.d_keypoints,self.d_model)
        
        self.positionnal_encoder = PositionalEncoding(self.d_model)

        self.class_token = nn.Parameter(torch.ones(1,self.d_model)) # no need 
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.mlp = nn.Linear(self.d_model,self.nb_actions)
    
    def forward(self,batch,batch_frames_valid ):
        """_summary_

        Args:
            batch (torch.tensor): of shape (N,T,P) where :
                        dimension 0 indexes the instance of the batch (detection on video)
                        dimension 1 indexes the frame (of the video)
                        dimension 2 indexes the keypoints (set of lamndmarks describins the detect person)
            batch_frames_valid (torch.tensor) : of shape (N,T) of bool type, where position (i,j)
                        indicates if object is detected or not on the frame
            
            frames from which no detection is present , are such that the correponding slice at batch
            is filled with np.nan, 
        """
        #convert keypoint positions to embedding space
        embeddings = self.embedding_layer(batch)

        # remove frames with no detection
        # remove the tokens

        class_token_duplicated = self.class_token.repeat(embeddings.shape[0],1,1)

        full_embeddings = torch.cat([class_token_duplicated,embeddings],1)

        # add positionnal encoding to the transformer modoel
        full_embeddings = self.positionnal_encoder(full_embeddings)

        src_key_padding_mask=~batch_frames_valid
        src_key_padding_mask = torch.concat([torch.zeros((src_key_padding_mask.shape[0],1),dtype=torch.bool),
                                             src_key_padding_mask],1)
        output = self.transformer_encoder(full_embeddings,src_key_padding_mask=src_key_padding_mask)

        prediction = self.mlp(output[:,0])
        return prediction
    

import math
class PositionalEncoding(nn.Module):
    """with batch as first index"""

    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1,max_len, d_model)
        pe[0,:, 0::2] = torch.sin(position * div_term)
        pe[0,:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self,x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)