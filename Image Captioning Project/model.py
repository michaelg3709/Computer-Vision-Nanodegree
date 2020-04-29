import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        # params
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
       
        # define layers
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True)
        self.hidden2out = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        
        # embedded word vectors for each token in a batch of captions
        captions = captions[:,:-1]
        embedded = self.word_embeddings(captions)
        
        inputs = torch.cat((features.unsqueeze(dim=1), embedded), dim=1)
        lstm_out, _ = self.lstm(inputs);
        
        outputs = self.hidden2out(lstm_out);
        
        return outputs
        
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption = []
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)        
            outputs = self.hidden2out(lstm_out.squeeze(1))       
            _, predicted = outputs.max(dim=1)                    
            caption.append(predicted.item())
            
            # next iteration
            inputs = self.word_embeddings(predicted.unsqueeze(1))    

        return caption
    
    
    