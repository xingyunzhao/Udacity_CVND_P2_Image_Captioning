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
        
        # embedding layer 
        # vocab_size: dimention of vacablary, before embedding
        # embed_size: the embedding size, also the input dimention to RNN
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM layer set up
        input_size = embed_size
        hidden_size = hidden_size   # output_size will be the same as hidden_size
        batch_first =True
        dropout = 0              
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first, dropout=dropout)

        # the linear layer that maps the hidden state output dimension 
        # to the vocab space
        self.linear_fc = nn.Linear(hidden_size, vocab_size)

        #         pass

    
    def forward(self, features, captions):
        
        # input parameters: 
        # features: feature vector from CNN encoder, it will be shape as (batch_size, embed_size)
        # captions: a list of captions, it will be shape as (batch_size, caption_length)
                
        # remove the <end> to ensure the length of caption equals to the output length of lstm, 
        # since the image feature will be concat to the beginning of catption
        captions = captions[:, :-1]
        
        # embedding captions, output dimention (batch, caption_length, embed_size)
        captions = self.word_embeddings(captions)
               
        # Concatenate the feature vectors for image and captions.
        # Image features shape: (batch_size, embed_size)
        # Word embeddings shape: (batch_size, caption_length-1, embed_size)
        # output shape : (batch_size, caption length, embed_size)
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        
        # Pass through the lstm, the output will be output and hidden state         
        # the output (h_n, c_n) is not used, so the returned value is denoted by _.
        # Input to LSTM : concatenated tensor(features, embeddings) and (h_0, c_0) default to zero
        # output shape : (batch_size, caption length, hidden_size)
        outputs, _ = self.lstm(inputs)
        
        # linearn fully connect layer to map output to vocab space
        # output shape : (batch_size, caption length, vocab_size)        
        outputs = self.linear_fc(outputs)        
        
        return outputs
        
        


    def sample(self, inputs, states=None, max_len=20):
        #" accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        outputs = []  # empty list to collect predicted words
        output_length = 0  # a counter to monitor the length of predicted sentense
        
        while (output_length <= max_len):
            
            # LSTM layer to get predicted word in embedding space
            # input  : (1, 1,embed_size) one batch, one word, embeded size
            # output : (1, 1,hidden_size) one batch, one word, hidden size
            # States should be passed to LSTM on each iteration in order for it to recall the last word it produced.
            output, states = self.lstm(inputs,states)
           
            ''' Linear layer '''
            # input  : (1,hidden_size)
            # output : (1,vocab_size)
            output = self.linear_fc(output.squeeze(dim = 1))  # convert embedding space to vocab space          
            _, predicted_index = torch.max(output, 1)  # pick the index of largest prob
           
            # append result
            outputs.append(predicted_index.cpu().numpy()[0].item())
            
            
            # checking if predicted index is 1, which means predict <end>. If <end>, then break loop even not reach the max length
            if (predicted_index == 1):
                break
            
            # Prepare for net loop iteration 
            # Embed the last predicted word to be the new input of the LSTM
            # To understand this step, again look at the diagram at end of  [ Notebook 1 ]
            inputs = self.word_embeddings(predicted_index)   
            inputs = inputs.unsqueeze(1)
            
            # To move to the next iteration of the while loop.
            output_length += 1
            
        return outputs
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        pass