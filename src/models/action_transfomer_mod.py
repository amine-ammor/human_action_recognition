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
        
        self.class_token = nn.Parameter(torch.ones(1,self.d_model)) # no need 
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.mlp = nn.Linear(self.d_model,self.nb_actions)
    def forward(self,batch):
        """_summary_

        Args:
            batch (torch.tensor): of shape (N,T,P) where :
                        dimension 0 indexes the instance of the batch (detection on video)
                        dimension 1 indexes the frame (of the video)
                        dimension 2 indexes the keypoints (set of lamndmarks describins the detect person)
        """
        embeddings = self.embedding_layer(batch)
        class_token_duplicated = self.class_token.repeat(embeddings.shape[0],1,1)

        transformer_input = torch.cat([class_token_duplicated,embeddings],1)
        output = self.transformer_encoder(transformer_input)

        prediction = self.mlp(output[:,0])
        return prediction