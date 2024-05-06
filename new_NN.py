import torch
import numpy as np

n1 = 10
n2 = 10
learning_rate=1e-4

def dec_tree(nabla, omega, nabla_crit, omega_crit):
    # Assuming nabla_crit and omega_crit are scalar values
    if nabla < nabla_crit and omega > omega_crit:
        return 1
    else:
        return 0

class MyNetwork(torch.nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc_sigma = torch.nn.Sequential(
            torch.nn.Linear(1, n1),  # Hidden layer 1 for sigma
            torch.nn.Sigmoid(),
            torch.nn.Linear(n1, n2),  # Hidden layer 2 for sigma
            torch.nn.Sigmoid(),
            torch.nn.Linear(n2, 2)  # Output layer for nabla_crit and omega_crit
        )
        self.fc_nabla = torch.nn.Linear(1, 1)  # Input nabla
        self.fc_omega = torch.nn.Linear(1, 1)  # Input omega

    def forward(self, sigma, x, y):
        sigma_hidden = self.fc_sigma(sigma)  # Pass sigma through hidden layers
        nabla_crit, omega_crit = torch.split(sigma_hidden, 1, dim=1)  # Split sigma into nabla_crit and omega_crit
        nabla_crit = torch.relu(nabla_crit)  # Apply ReLU to ensure positive values
        omega_crit = torch.relu(omega_crit)
        nabla_out = torch.relu(self.fc_nabla(x))  # Pass x (nabla) through its layer
        omega_out = torch.relu(self.fc_omega(y))  # Pass y (omega) through its layer
        
        # Pass inputs to decision tree function
        output = dec_tree(nabla_out, omega_out, nabla_crit, omega_crit)
        return output

# Create an instance of the network
model = MyNetwork()
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #lr values usually about 10^-3 - 10^-5

data_path_temp = 'nablatemp-slice-B1-0000080000.raw'
data_path_reaction = 'wtemp-slice-B1-0000080000.raw'

data_temp = np.fromfile(data_path_temp, count=-1, dtype=np.float64)
data_reaction = np.fromfile(data_path_reaction, count=-1, dtype=np.float64)



for i in range(len(data_temp)):
    phi_pred=model(data_temp, data_reaction, 0)
    loss=loss_fn()
    optimizer.zero_grad()
    