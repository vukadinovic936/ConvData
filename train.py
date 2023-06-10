import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm

class FCN(nn.Module):
    def __init__(self, d=28):
        super().__init__()
        #W
        self.lin1 = nn.Linear(28*28,100)
        #V fix it to 1/100
        self.lin2 = nn.Linear(1,100)
        self.lin2.weight.data=torch.ones(1,100)/100
        self.lin2.bias.data = torch.zeros(1)
        #self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.lin1(x)
        x = F.sigmoid(x)
        x = torch.erf(x / torch.sqrt(torch.tensor(2.0)))
        x = self.lin2(x)
        return x
def IPR(w):
    return torch.sum(w**4)/(torch.sum(w**2)**2)
def visualize_fields(W):
    """
    W - K x (NxN) matrix
        K number of linear units
        NxN shape of the image
    """
    K,N_squared = W.shape
    N = int(np.sqrt(N_squared))
    for i in range(K):
        print(IPR(W[i]))
        plt.imshow(W[i].reshape((N,N)))
        plt.show()
import numpy as np
# define functions
vec_erf = np.vectorize(math.erf)
psi = lambda z: vec_erf(z/(np.sqrt(2)))

gain_factor=5
normalizer = np.sqrt(2/np.pi * np.arcsin(gain_factor**2/(1+gain_factor**2)))
epsilon_plus=5.6
epsilon_minus=2.8 
mean_vector = np.zeros(28)
positive_cov_matrix = np.ones((28,28))
negative_cov_matrix = np.ones((28,28))

# Generate a random vector
#random_vector = np.random.multivariate_normal(mean_vector, cov_matrix)
for i in range(28):
    for j in range(28):
        positive_cov_matrix[i,j] = np.exp(- ( np.abs(i-j) / epsilon_plus)**2 )
positive_cov_matrix=np.kron(positive_cov_matrix,positive_cov_matrix)

for i in range(28):
    for j in range(28):
        negative_cov_matrix[i,j] = np.exp(- ( np.abs(i-j) / epsilon_minus)**2 )
negative_cov_matrix=np.kron(negative_cov_matrix,negative_cov_matrix)

def get_online_batch(n=1000):
    #np.random.random()
    positive_size=int(0.5*n)
    negative_size=n-positive_size
    
    z_mu_pos = np.random.multivariate_normal(np.ones(784), positive_cov_matrix,positive_size)
    positive_dataset = psi(gain_factor * z_mu_pos)/normalizer
    
    z_mu_neg = np.random.multivariate_normal(np.ones(784), negative_cov_matrix,negative_size)
    negative_dataset = psi(gain_factor * z_mu_neg)/normalizer
    
    batch = torch.from_numpy(np.vstack([negative_dataset, positive_dataset]).astype(np.float32))
    batch_labels = torch.from_numpy(np.array([-1]*negative_size + [1]*positive_size).astype('float32'))
    # shuffle 
    shuf_idx= torch.randperm(n)
    batch=batch[shuf_idx]
    batch_labels=batch_labels[shuf_idx]
    return batch, batch_labels

## Train the net
fc = FCN()
mse = torch.nn.MSELoss()
optimizer = torch.optim.SGD(fc.parameters(), lr=0.05, momentum=0.9)
#optimizer = torch.optim.Adam(fc.parameters(), lr=0.001)
num_epochs = 10000

def total_IPR(W):
    return (torch.sum(W**4,dim=1)/(torch.sum(W**2, dim=1))**2)

save_IPR=[]

for epoch in tqdm(range(num_epochs)):
    
    inputs, labels = get_online_batch(1000)
    out = fc(inputs)
    loss = mse(out.squeeze(), labels)
    
    ## Backwards pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch%10==0:
        with open('MSE.txt', 'a') as f:
            f.write(f'EPOCH {epoch}: Train MSE is {loss}\n')
        with torch.no_grad():

            test_dataset, test_labels=get_online_batch(10000)
            test_outs = fc(test_dataset)
            accuracy = torch.sum( test_outs.sign().squeeze() == test_labels)/len(test_labels)
            with open("accuracy.txt", 'a') as f:
                f.write(f"EPOCH {epoch}: Accuracy is {accuracy}\n")

    if epoch%100==0:
        save_IPR.append(total_IPR(fc.lin1.weight.data).detach().numpy())
save_IPR=np.array(save_IPR)
np.save('ipr_list.npy',save_IPR)
torch.save(fc.state_dict(),"checkpoint.pth")            
print('Finished Training')