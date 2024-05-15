import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import torch
from sklearn.utils import shuffle
from comparison import calculate_MSE
from data_preparation import load_phi_field_NN_new

# spatial constants
nx_B1, ny_B1 = 384, 384
nx_A, ny_A = 512, 512
lx, ly = 0.01, 0.01  # [m]
dx, dy = lx / (nx_B1 - 1), ly / (ny_B1 - 1)
x = np.linspace(0, lx, nx_B1)
y = np.linspace(0, ly, ny_B1)

# flame related constants
TU_B1 = 1500.000  # K
TB_B1 = 1623.47  # K
DTH_B1 = 0.0012904903  # m
DU_B1 = 0.2219636  # kg/m^3
SL_B1 = 1.6585735551  # m/s

DTH_A = 0.00099990386 #m
DU_A = 0.2235816 #kg/m^3
SL_A = 2.6185663 #m/s
TU_A = 1500.000 #K
TB_A = 1691.876 #K

# normalization constants
CT_NORM_B1 = TB_B1 - TU_B1
WCT_NORM_B1 = DU_B1 * SL_B1 / DTH_B1
NCT_NORM_B1 = 1.0 / DTH_B1

CT_NORM_A = TB_A - TU_A
WCT_NORM_A = DU_A * SL_A / DTH_A
NCT_NORM_A = 1.0 / DTH_A

omega_plus_max_B1= 1996.8891/WCT_NORM_B1
nabla_T_plus_max_B1 = 3931.0113/NCT_NORM_B1
omega_plus_max_A1 = 3467.06161/WCT_NORM_A
nabla_T_plus_max_A1 = 4447.89830/NCT_NORM_A
omega_plus_max_A2 = 3339.96234/WCT_NORM_A
nabla_T_plus_max_A2 = 3250.13781/NCT_NORM_A

# standard deviation for gaussian filter
fwidth_n_B1 = np.array(['000', '025', '037', '049', '062', '074', '086', '099'])
fwidth_n_A = np.array(['000', '026', '038', '051', '064', '077', '089', '102'])
filter_sizes=np.array([0,0.5,0.75, 1.00, 1.25, 1.50, 1.75, 2.0])

def get_sig(filter_index, case): 
  if case=="A1" or case=="A2":
    sig = np.sqrt(int(fwidth_n_A[filter_index]) ** 2 / 12.0)
  else:
    sig = np.sqrt(int(fwidth_n_B1[filter_index]) ** 2 / 12.0)
  return sig

#filepaths
cases=['A1', 'A2', 'B1']
timesteps_A=["65", "80", "95"]

nabla_T_total=[]
omega_total=[]
phi_res_total=[]

#discretization/filtering params
reso_dis=65
nabla_T_omega_sig=2 #idk what an appropriate value for this is (ig just print as you go along)

# NN training params
epochs=200
batch_size=32
learning_rate0=5e-5
learning_rate1=8.5e-4 #dont go above 1e-3 5e-4 worked well 0.0048 or smth like that (8e-4 seems worse)
learning_rate2=3e-7
learning_rate3=5e-3
learning_rate4=8e-9

input_size = 3
n_hidden1 = 40  # Number of neurons in the first hidden layer
n_hidden2 = 40  # Number of neurons in the second hidden layer
n_hidden3 = 40  # Number of neurons in the third hidden layer
output_size = 1
# NN instantiation

class MyNeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, n_hidden1, n_hidden2, n_hidden3, output_size, dropout_prob=0.0003):
        super(MyNeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, n_hidden1, dtype=torch.float64)
        self.dropout1 = torch.nn.Dropout(p=dropout_prob)  # Dropout layer after the first hidden layer
        self.fc2 = torch.nn.Linear(n_hidden1, n_hidden2,dtype=torch.float64)
        self.dropout2 = torch.nn.Dropout(p=dropout_prob)  # Dropout layer after the second hidden layer
        self.fc3 = torch.nn.Linear(n_hidden2, n_hidden3, dtype=torch.float64)
        self.dropout3 = torch.nn.Dropout(p=dropout_prob)  # Dropout layer after the third hidden layer
        self.fc4 = torch.nn.Linear(n_hidden3, output_size, dtype=torch.float64)
        self.sigmoid = torch.nn.Sigmoid()  # Sigmoid activation function (can be replaced with other activations)

    def forward(self, x, training=True):
        out = self.fc1(x)
        out = self.dropout1(out) if training else out  # Apply dropout only during training
        out = self.sigmoid(out)

        out = self.fc2(out)
        out = self.dropout2(out) if training else out
        out = self.sigmoid(out)

        out = self.fc3(out)
        out = self.dropout3(out) if training else out
        out = self.sigmoid(out)

        out = self.fc4(out)
        return out


# Create an instance of the neural network
model = MyNeuralNetwork(input_size, n_hidden1, n_hidden2, n_hidden3, output_size)

loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate1) 

def f_exclude_boundary(filter_index, case):
  if case=="A1" or case=="A2":
    actual_filter_size = int(fwidth_n_A[filter_index])
  elif case=="B1":
    actual_filter_size = int(fwidth_n_B1[filter_index])
  # Exclusion boundaries
  base_exclusion_left = 25
  base_exclusion_right = 0
  additional_exclusion = 0.5 * actual_filter_size  # Adjust according to cell size if needed

  left_exclusion = base_exclusion_left + additional_exclusion
  right_exclusion = base_exclusion_right + additional_exclusion
  return int(left_exclusion), int(right_exclusion)

def exclude_boundaries(field, left_exclusion, right_exclusion):
    # Transpose and flip operations to match your data formatting needs
    field = np.flipud(field.T)
    # Only modify the second dimension (columns) of the array.
    if left_exclusion > 0 and right_exclusion > 0:
        # Print shape of the field before exclusion
        #print(f"Shape of field before exclusion: {field.shape}")
        cropped_field = field[:, left_exclusion:-right_exclusion or None]
        # Print shape of the field after exclusion
        #print(f"Shape of field after exclusion: {cropped_field.shape}")
        return cropped_field
    elif left_exclusion > 0:
        return field[:, left_exclusion:]
    elif right_exclusion > 0:
        return field[:, :-right_exclusion or None]
    else:
        return field

# getting data
def get_fields(case, timestep, filter_index):
    #select the correct file
    if filter_index==0:
        filepath_wtemp=r"Data_new_NN\dataset_slice_{}_TS{}\wtemp-slice-{}-00000{}000.raw".format(case, timestep, case, timestep)
        if case=="A1" or case=="A2":
            filepath_nabla_T=r"Data_new_NN\dataset_slice_{}_TS{}\tilde-nablatemp-slice-{}-00000{}000-{}.raw".format(case, timestep, case, timestep, fwidth_n_A[1])
        elif case=="B1":
            filepath_nabla_T=r"Data_new_NN\dataset_slice_{}_TS{}\nablatemp-slice-{}-00000{}000.raw".format(case, timestep, case, timestep)
        else:
            print("1 - what case this", case)
        #todo - add the actual unfiltered nabla_T path when done, and remove the if statement above (if proper implementation, it
        #is unnecessary)
    else:
        if case=="A1" or case=="A2":
            filepath_wtemp=r"Data_new_NN\dataset_slice_{}_TS{}\bar-wtemp-slice-{}-00000{}000-{}.raw".format(case, timestep, case, timestep, fwidth_n_A[filter_index])
            filepath_nabla_T=r"Data_new_NN\dataset_slice_{}_TS{}\tilde-nablatemp-slice-{}-00000{}000-{}.raw".format(case, timestep, case, timestep, fwidth_n_A[filter_index])
        elif case=="B1":
            filepath_wtemp=r"Data_new_NN\dataset_slice_{}_TS{}\bar-wtemp-slice-{}-00000{}000-{}.raw".format(case, timestep, case, timestep, fwidth_n_B1[filter_index])
            filepath_nabla_T=r"Data_new_NN\dataset_slice_{}_TS{}\tilde-nablatemp-slice-{}-00000{}000-{}.raw".format(case, timestep, case, timestep, fwidth_n_B1[filter_index])
        else:
            print("2 - what case this", case)
    
    #select correct sizing and proper normalization
    if case=="A1" or case=="A2":
        wtemp=np.fromfile(filepath_wtemp, count=-1, dtype=np.float64).reshape(nx_A, ny_A)
        nabla_T=np.fromfile(filepath_nabla_T, count=-1, dtype=np.float64).reshape(nx_A, ny_A)
        omega_bar_plus = wtemp / (CT_NORM_A * WCT_NORM_A)
        nabla_T_bar_plus = nabla_T / (CT_NORM_A * NCT_NORM_A)
    elif case=="B1":
        wtemp=np.fromfile(filepath_wtemp, count=-1, dtype=np.float64).reshape(nx_B1, ny_B1)
        nabla_T=np.fromfile(filepath_nabla_T, count=-1, dtype=np.float64).reshape(nx_B1, ny_B1)
        omega_bar_plus = wtemp / (CT_NORM_B1 * WCT_NORM_B1)
        nabla_T_bar_plus = nabla_T/ (CT_NORM_B1 * NCT_NORM_B1)

    left_exclude, right_exclude=f_exclude_boundary(filter_index,case)

    nabla_T_bar_plus=exclude_boundaries(nabla_T_bar_plus, left_exclude, right_exclude)
    omega_bar_plus = exclude_boundaries(omega_bar_plus, left_exclude, right_exclude)

    return omega_bar_plus, nabla_T_bar_plus

#save unfiltered res so that we do not need to do all the preprocessing for every filter size

def process_res_unfiltered():
    for case in cases:
        if case=="A1" or case=="A2":
            for timestep in timesteps_A:
                omega_plus, nabla_T_plus=get_fields(case, timestep, 0) 
                phi = np.zeros_like(omega_plus)
                if case=="A1":
                    omega_star=omega_plus/omega_plus_max_A1
                    nabla_T_star=nabla_T_plus/nabla_T_plus_max_A1
                elif case=="A2":
                    omega_star=omega_plus/omega_plus_max_A2
                    nabla_T_star=nabla_T_plus/nabla_T_plus_max_A2
                phi[(omega_star > 0.4) & (nabla_T_star < 0.2)] = 1   
                np.save(r"unfiltered_res\unfiltered_res-{}-{}.npy".format(case, timestep), phi)

        else:
            omega_plus, nabla_T_plus=get_fields(case, "80", 0)
            omega_star=omega_plus/omega_plus_max_B1
            nabla_T_star=nabla_T_plus/nabla_T_plus_max_B1
            phi = np.zeros_like(omega_plus)
            phi[(omega_star > 0.4) & (nabla_T_star < 0.2)] = 1
            np.save(r"unfiltered_res\unfiltered_res-{}-80.npy".format(case), phi)
#process_res_unfiltered() #note that to do this, you need to comment out the exclude boundaries of get_phi_res and get_fields

def get_phi_res(case, timestep, filter_index):
    sig=get_sig(filter_index, case)
    phi_unfiltered = np.load(r"unfiltered_res\unfiltered_res-{}-{}.npy".format(case, timestep))
    left_exclude, right_exclude=f_exclude_boundary(filter_index,case)
    phi_unfiltered=exclude_boundaries(phi_unfiltered, left_exclude, right_exclude)
    phi_res = scipy.ndimage.gaussian_filter(phi_unfiltered, sigma=sig)
    return phi_res

#testing that phi_res works
"""
omega_bar_plus, nabla_T_bar_plus=get_fields("B1", "80",1)
phi_res=get_phi_res("B1", "80",1)
fig, ax=plt.subplots()
scatter_pred=ax.scatter(omega_bar_plus, nabla_T_bar_plus, c=phi_res)
plt.show()

phi_res=get_phi_res("B1", "80",5)
fig, ax=plt.subplots()
exclude_boundaries_L, exclude_boundaries_R=f_exclude_boundary(5, "B1")
# Define grid dimensions explicitly based on phi_res shape
scatter_pred=ax.pcolor(np.flipud(phi_res), cmap='jet')
plt.colorbar(scatter_pred)
plt.show()
"""

#getting discretized arrays for phi_res
def get_discrete_phi_res(case, filter_index, timestep, res):
    freq_array=np.zeros((res,res))
    phi_res_aggregate=np.zeros((res,res))
    phi_res_average=np.zeros((res,res))

    omega_bar_plus, nabla_T_bar_plus=get_fields(case, timestep,filter_index)
    phi_res=get_phi_res(case, timestep,filter_index)

    nabla_T_bar_plus_range=(0,nabla_T_bar_plus.max()*(1+1/res))
    omega_bar_plus_range=(0,omega_bar_plus.max()*(1+1/res))
    omega_bar_plus_discrete = np.arange(omega_bar_plus_range[0], omega_bar_plus_range[1], (omega_bar_plus_range[1]-omega_bar_plus_range[0])/res)
    nabla_T_bar_plus_discrete = np.arange(nabla_T_bar_plus_range[0], nabla_T_bar_plus_range[1],  (nabla_T_bar_plus_range[1]-nabla_T_bar_plus_range[0])/res)
    omega_bar_plus_grid, nabla_T_bar_plus_grid = np.meshgrid(omega_bar_plus_discrete, nabla_T_bar_plus_discrete)
    actual_dis_phi_vals=[]
    actual_dis_nabla_T_bar_plus_vals=[]
    actual_dis_omega_bar_plus_vals=[]

    omega_bar_plus_flat=omega_bar_plus.flatten()
    nabla_T_bar_plus_flat=nabla_T_bar_plus.flatten()
    phi_res_flat=phi_res.flatten()

    for i in range(len(omega_bar_plus_flat)):
        ind_omega=int((omega_bar_plus_flat[i]*res)/(omega_bar_plus_range[1]-omega_bar_plus_range[0])+0.5)
        ind_nabla_T=int((nabla_T_bar_plus_flat[i]*res)/(nabla_T_bar_plus_range[1]-nabla_T_bar_plus_range[0])+0.5)
        freq_array[ind_nabla_T][ind_omega]+=1
        phi_res_aggregate[ind_nabla_T][ind_omega] +=phi_res_flat[i]
    for i in range(len(freq_array)):
        for j in range(len(freq_array[i])):
            if freq_array[i,j]!=0:
                phi_res_average[i,j]=phi_res_aggregate[i,j]/freq_array[i,j]
                actual_dis_phi_vals.append(phi_res_average[i,j])
                actual_dis_nabla_T_bar_plus_vals.append(nabla_T_bar_plus_grid[i,j])
                actual_dis_omega_bar_plus_vals.append(omega_bar_plus_grid[i,j])
    
    actual_dis_phi_vals=np.array(actual_dis_phi_vals)
    actual_dis_nabla_T_bar_plus_vals=np.array(actual_dis_nabla_T_bar_plus_vals)
    actual_dis_omega_bar_plus_vals=np.array(actual_dis_omega_bar_plus_vals)

    return actual_dis_omega_bar_plus_vals, actual_dis_nabla_T_bar_plus_vals, actual_dis_phi_vals

#load model (IMPORTANT - if this is dissabled, you will lose the model and have to retrain it from scratch)
def load_model():
    global model
    responded=False
    while not responded:
        decision=input("Do you wish to load the model? IMPORTANT - if you respond no, the model will have to be retrained from scratch.\n y/n: ")
        if decision == 'y':
            #model = torch.load("new_model_discretized.pt")
            model=torch.load("Models_During_Training\\new_model_discretized40.pt")
            responded=True
        elif decision=='n':
            responded=True
        else:
            print("\'"+decision+"\' is not a valid respose, please input \'y\' or \'n\'." )
load_model()

#trying out model to understand characteristic function
def try_nn( filter_index, real, timestep="80", case="B1", reso_dis=200):
    if real:
        actual_dis_omega_bar_plus_vals, actual_dis_nabla_T_bar_plus_vals, actual_dis_phi_vals=get_discrete_phi_res(case, filter_index, timestep, 100)
        omega_bar_plus_flat=actual_dis_omega_bar_plus_vals.flatten()
        nabla_T_bar_plus_flat=actual_dis_nabla_T_bar_plus_vals.flatten()
        phi_res_flat=actual_dis_phi_vals.flatten()
        fig, ax=plt.subplots(2)
        scatter_actual=ax[1].scatter(omega_bar_plus_flat, nabla_T_bar_plus_flat, c=phi_res_flat, s=50/reso_dis)
    else:
        actual_dis_omega_bar_plus_vals=np.linspace(0, max(omega_plus_max_A1, omega_plus_max_A2, omega_plus_max_B1), reso_dis)
        actual_dis_nabla_T_bar_plus_vals=np.linspace(0, max(nabla_T_plus_max_A1, nabla_T_plus_max_A2, nabla_T_plus_max_B1), reso_dis)
        omega_bar_plus_grid, nabla_T_bar_plus_grid=np.meshgrid(actual_dis_omega_bar_plus_vals, actual_dis_nabla_T_bar_plus_vals)
        omega_bar_plus_flat=omega_bar_plus_grid.flatten()
        nabla_T_bar_plus_flat=nabla_T_bar_plus_grid.flatten()
        fig, ax=plt.subplots()
        
    phi_preds=[]
    for i in range(len(omega_bar_plus_flat)):
        inputs=[omega_bar_plus_flat[i], nabla_T_bar_plus_flat[i], filter_sizes[filter_index]]
        inputs_tensor=torch.tensor(inputs, dtype=torch.float64)
        phi_pred=model(inputs_tensor,training=False)
        phi_preds.append(phi_pred.detach().numpy())

        if i%500==0:
            print('running model', i)

    phi_preds=np.array(phi_preds)
    if real:
        scatter_pred=ax[0].scatter(omega_bar_plus_flat, nabla_T_bar_plus_flat, c=phi_preds, s=50/reso_dis)
    else:
        scatter_pred=ax.scatter(omega_bar_plus_flat, nabla_T_bar_plus_flat, c=phi_preds, s=50/reso_dis)
    plt.show()

def get_theoretical_vals():
    for filter_size in filter_sizes:
        res=500
        nabla_T_range=(0,max(nabla_T_plus_max_A1, nabla_T_plus_max_A2, nabla_T_plus_max_B1))
        omega_range=(0,max(omega_plus_max_A1, omega_plus_max_A2, omega_plus_max_B1))

        omega_discrete = np.arange(omega_range[0], omega_range[1], (omega_range[1]-omega_range[0])/res)
        nabla_T_discrete = np.arange(nabla_T_range[0], nabla_T_range[1],  (nabla_T_range[1]-nabla_T_range[0])/res)
        grid_omega, grid_nabla_T = np.meshgrid(omega_discrete, nabla_T_discrete)
        omega_bar_plus_flat=grid_omega.flatten()
        nabla_T_bar_plus_flat=grid_nabla_T.flatten()
        phi_NN_field=[]
        for i in range(len(omega_bar_plus_flat)):
            inputs=[omega_bar_plus_flat[i], nabla_T_bar_plus_flat[i], filter_size/2]
            inputs_tensor=torch.tensor(inputs, dtype=torch.float64)
            phi_pred=model(inputs_tensor, training=False)
            phi_NN_field.append(phi_pred.detach().numpy())

        phi_NN_field=np.array(phi_NN_field).reshape(grid_omega.shape)
        np.save(f"NewNNPredTheoretical\\Array{filter_size}.npy", phi_NN_field)
        print(filter_size)
        plt.pcolor(grid_omega/omega_plus_max_B1, grid_nabla_T/nabla_T_plus_max_B1, phi_NN_field, rasterized=True)
        plt.xlabel('$\\overline{\\omega}_{c_{T}}^*$', fontsize=25)
        plt.ylabel('$| \\nabla \\tilde{c}_{T}|^*$', fontsize=25)
        plt.colorbar()
        plt.savefig(f"NewNNPredTheoretical\\{filter_size}.pdf", bbox_inches="tight")
        plt.close()

def get_model_plots():
    for case in cases:
        for timestep in timesteps_A:
            if case=="B1" and timestep!="80":
                continue
            else:
                min_index=1
                if case=="B1": #todo - if i get unfiltered nabla vals for case A, change this
                    min_index=0
                for filter_index in range(min_index, len(filter_sizes)):
                    print("Plotting",case, timestep, filter_sizes[filter_index])
                    actual_dis_omega_bar_plus_vals, actual_dis_nabla_T_bar_plus_vals, actual_dis_phi_vals=get_discrete_phi_res(case, filter_index, timestep, 100)
                    omega_bar_plus_flat=actual_dis_omega_bar_plus_vals.flatten()
                    nabla_T_bar_plus_flat=actual_dis_nabla_T_bar_plus_vals.flatten()
                    phi_res_flat=actual_dis_phi_vals.flatten()
                    filter_size=filter_sizes[filter_index]/2
                    phi_preds=[]
                    
                    for i in range(len(omega_bar_plus_flat)):
                        inputs=[omega_bar_plus_flat[i], nabla_T_bar_plus_flat[i], filter_size]
                        inputs_tensor=torch.tensor(inputs, dtype=torch.float64)
                        phi_pred=model(inputs_tensor, training=False)
                        phi_preds.append(phi_pred.detach().numpy())

                    phi_preds=np.array(phi_preds)
                    fig, ax=plt.subplots(2)
                    scatter_pred=ax[0].scatter(omega_bar_plus_flat, nabla_T_bar_plus_flat, c=phi_preds, s=80/reso_dis)
                    scatter_actual=ax[1].scatter(omega_bar_plus_flat, nabla_T_bar_plus_flat, c=phi_res_flat, s=80/reso_dis)
                    plt.savefig(r"ModelPreds\Filter{}\{}-{}.jpeg".format(filter_size*2, case, timestep))
                    plt.close()

#training model, including different timesteps, cases, 
"""

for epoch in range(epochs):
    if epoch==20:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate2) 
    for case in cases:
        for timestep in timesteps_A:
            if case=="B1" and timestep!="80":
                continue
            else:
                min_index=1
                if case=="B1": #todo - if i get unfiltered nabla vals for case A, change this
                    min_index=0
                for filter_index in range(min_index, len(filter_sizes)):
                    print(case, timestep, filter_sizes[filter_index], epoch+1)
                    actual_dis_omega_bar_plus_vals, actual_dis_nabla_T_bar_plus_vals, actual_dis_phi_vals=get_discrete_phi_res(case, filter_index, timestep)
                    omega_bar_plus_flat=actual_dis_omega_bar_plus_vals.flatten()
                    nabla_T_bar_plus_flat=actual_dis_nabla_T_bar_plus_vals.flatten()
                    phi_res_flat=actual_dis_phi_vals.flatten()
                    omega_bar_plus_flat_shuffled, nabla_T_bar_plus_flat_shuffled, phi_res_flat_shuffled = shuffle(
                    omega_bar_plus_flat, nabla_T_bar_plus_flat, phi_res_flat, random_state=3)
                    filter_size=filter_sizes[filter_index]/2
                    phi_preds=[]
                    print(len(omega_bar_plus_flat))

                    #validation before adaption, otherwise you always see how it has adapted after it has adapted - cheating a bit
                    for i in range(len(omega_bar_plus_flat)):
                        inputs=[omega_bar_plus_flat[i], nabla_T_bar_plus_flat[i], filter_size]
                        inputs_tensor=torch.tensor(inputs, dtype=torch.float64)
                        phi_pred=model(inputs_tensor, training=False)
                        phi_preds.append(phi_pred.detach().numpy())

                        if i%500==0:
                            print('running model', i)

                    phi_preds=np.array(phi_preds)
                    fig, ax=plt.subplots(2)
                    scatter_pred=ax[0].scatter(omega_bar_plus_flat, nabla_T_bar_plus_flat, c=phi_preds, s=80/reso_dis)
                    scatter_actual=ax[1].scatter(omega_bar_plus_flat, nabla_T_bar_plus_flat, c=phi_res_flat, s=80/reso_dis)
                    plt.savefig(r"ModelPreds\Filter{}\{}-{}.jpeg".format(filter_size*2, case, timestep))
                    plt.close()

                    #adapting
                    repeat_times=1
                    if filter_index==0: #I want to make it understand the unfiltered case better
                        repeat_times=2
                    for i in range(repeat_times):
                        for i in range(len(omega_bar_plus_flat)):
                            inputs=[omega_bar_plus_flat_shuffled[i], nabla_T_bar_plus_flat_shuffled[i], filter_size]
                            inputs_tensor=torch.tensor(inputs, dtype=torch.float64)
                            phi_pred=model(inputs_tensor)

                            loss = loss_fn(phi_pred, torch.tensor([phi_res_flat_shuffled[i]], dtype=torch.float64))
                            if i%500==0:
                                print(i)
                
                            # Backpropagation
                            optimizer.zero_grad()
                            loss.backward() #calculates the gradients
                            optimizer.step()

                    torch.save(model, "new_model_discretized")
"""

#second training function - since the first one has some 'biases'
#this one is good, but i just dont want to train atm

def get_all_discretized_data(res):
    all_dis_nabla_T_bar_plus=[]
    all_dis_omega_bar_plus=[]
    all_dis_phi_res=[]
    all_filter_sizes=[]
    for case in cases:
            for timestep in timesteps_A:
                if case=="B1" and timestep!="80":
                    continue
                else:
                    min_index=1
                    if case=="B1": #todo - if i get unfiltered nabla vals for case A, change this
                        min_index=0
                    for filter_index in range(min_index, len(filter_sizes)):
                        repeat_times=1
                        if filter_index==0: #I want to make it understand the unfiltered case better
                            repeat_times=2
                        for i in range(repeat_times):
                            print(case, timestep, filter_sizes[filter_index])
                            actual_dis_omega_bar_plus_vals, actual_dis_nabla_T_bar_plus_vals, actual_dis_phi_vals=get_discrete_phi_res(case, filter_index, timestep, res)
                            omega_bar_plus_flat=actual_dis_omega_bar_plus_vals.flatten()
                            nabla_T_bar_plus_flat=actual_dis_nabla_T_bar_plus_vals.flatten()
                            phi_res_flat=actual_dis_phi_vals.flatten()
                            filter_size=filter_sizes[filter_index]/2
                            for i in range(len(omega_bar_plus_flat)):
                                all_dis_nabla_T_bar_plus.append(nabla_T_bar_plus_flat[i])
                                all_dis_omega_bar_plus.append(omega_bar_plus_flat[i])
                                all_dis_phi_res.append(phi_res_flat[i])
                                all_filter_sizes.append(filter_size)
    all_dis_nabla_T_bar_plus=np.array(all_dis_nabla_T_bar_plus)
    all_dis_omega_bar_plus=np.array(all_dis_omega_bar_plus)
    all_dis_phi_res=np.array(all_dis_phi_res)
    all_filter_sizes=np.array(all_filter_sizes)
    np.save(f"Discretized_Data\\all_dis_nabla_T_bar_plus{res}.npy", all_dis_nabla_T_bar_plus)
    np.save(f"Discretized_Data\\all_dis_omega_bar_plus{res}.npy", all_dis_omega_bar_plus)
    np.save(f"Discretized_Data\\all_dis_phi_res{res}.npy", all_dis_phi_res)
    np.save(f"Discretized_Data\\all_filter_sizes{res}.npy", all_filter_sizes)

def load_discretized_data(res):
    all_dis_nabla_T_bar_plus=np.load(f"Discretized_Data\\all_dis_nabla_T_bar_plus{res}.npy")
    all_dis_omega_bar_plus=np.load(f"Discretized_Data\\all_dis_omega_bar_plus{res}.npy")
    all_dis_phi_res=np.load(f"Discretized_Data\\all_dis_phi_res{res}.npy")
    all_filter_sizes=np.load(f"Discretized_Data\\all_filter_sizes{res}.npy")
    return all_dis_nabla_T_bar_plus, all_dis_omega_bar_plus, all_dis_phi_res, all_filter_sizes

def train_NN():
    MSE_vals=[[1,1,1,1,1,1,1]]
    MSE_vals=np.array(MSE_vals)
    #res=30
    res=65
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate1)
    all_dis_nabla_T_bar_plus, all_dis_omega_bar_plus, all_dis_phi_res, all_filter_sizes=load_discretized_data(res)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate1) 
    MSE_for_epoch=np.array([])
    for epoch in range(epochs):
        if epoch==4:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate3)
            #res=60
            #all_dis_nabla_T_bar_plus, all_dis_omega_bar_plus, all_dis_phi_res, all_filter_sizes=load_discretized_data(res)
        #if epoch==int(epochs*0.7):
            #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate3)
            #res=120
            #all_dis_nabla_T_bar_plus, all_dis_omega_bar_plus, all_dis_phi_res, all_filter_sizes=load_discretized_data(res)
        array_mean=np.mean(MSE_for_epoch)
        if array_mean<0.009:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate1)
        if array_mean<0.006:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate2)
        if array_mean<0.0035:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate4)
        print(array_mean, optimizer)
        MSE_for_epoch=np.array([])
        omega_bar_plus_flat_shuffled, nabla_T_bar_plus_flat_shuffled, phi_res_flat_shuffled, filter_sizes_shuffled = shuffle(
                        all_dis_omega_bar_plus,all_dis_nabla_T_bar_plus, all_dis_phi_res, all_filter_sizes, random_state=epoch)
        print(len(omega_bar_plus_flat_shuffled))
        for i in range(len(omega_bar_plus_flat_shuffled)):
            inputs=[omega_bar_plus_flat_shuffled[i], nabla_T_bar_plus_flat_shuffled[i], filter_sizes_shuffled[i]]
            inputs_tensor=torch.tensor(inputs, dtype=torch.float64)
            phi_pred=model(inputs_tensor)

            loss = loss_fn(phi_pred, torch.tensor([phi_res_flat_shuffled[i]], dtype=torch.float64))
            # Backpropagation
            optimizer.zero_grad()
            loss.backward() #calculates the gradients
            optimizer.step()
            
            if i%10000==0:
                print("Epoch "+str(epoch+1), i)
        torch.save(model, r'Models_During_Training\\new_model_discretized{}.pt'.format(epoch))# todo- just changed the name, remember to change it to the actual file later (and push)
        load_phi_field_NN_new(epoch)
        for filter in filter_sizes[1:]:
            MSE=calculate_MSE(filter, "newNN")
            MSE_for_epoch=np.append(MSE_for_epoch, MSE)
        MSE_vals=np.append(MSE_vals, MSE_for_epoch)
        print(MSE_for_epoch, epoch)
        np.save("MSE_vals_training", MSE_vals)
    get_model_plots()



#run functions here

#get_all_discretized_data(30)
#get_all_discretized_data(60)
#get_all_discretized_data(100)
#get_all_discretized_data(65)

#train_NN()

get_theoretical_vals()
#get_model_plots()

