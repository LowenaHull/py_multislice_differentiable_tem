#!/usr/bin/env python
# coding: utf-8

# # Protein Optimisation

# In[1]:


import pyms
import numpy as np
import torch
import time
from pyms.Probe import aberration
from copy import deepcopy
import matplotlib.pyplot as plt
import math
import os
from datetime import date
from abtem.transfer import CTF
from ase.io.proteindatabank import read_proteindatabank
from ase.io import read
import abtem
from ase.visualize import view

from pyms.utils.torch_utils import (
    amplitude,
    get_device,
    crop_to_bandwidth_limit_torch,
    size_of_bandwidth_limited_array,
    torch_dtype_to_numpy,
    real_to_complex_dtype_torch,
    ensure_torch_array
)

from pyms.Probe import (
    make_contrast_transfer_function,
)

from pyms.utils.numpy_utils import (
    fourier_shift,
    crop_to_bandwidth_limit,
    ensure_array,
    q_space_array,
)


get_ipython().run_line_magic('matplotlib', 'inline')

dps = read_proteindatabank('/home/hremadmin/Downloads/6zgl.pdb')

dps.center(vacuum = 30, axis = (0,1))
dps.center(vacuum = 1, axis = (2))

dps_pyms = pyms.structure_routines.structure.from_ase_cluster(dps)

# Subslicing of crystal for multislice
subslices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# Grid size in pixels
gridshape = [1024, 1024]
thicknesses= dps_pyms.unitcell[2]+1
eV = 1e5
tiling = [1,1]


# In[2]:


def create_working_dir():
    directory = "Run_1"
    parent_dir = '/home/hremadmin/Documents/Project/ML_files/Experimental_Data_2nd/'
    path = os.path.join(parent_dir, directory)
    
    if os.path.isdir(path):
        i = 2
        while os.path.isdir(path):
            directory = "Run_" + str(i)
            path = os.path.join(parent_dir, directory)
            i += 1
    os.mkdir(path)
    
    return path


# In[3]:


def create_folders(working_dir):
    
    directory = "Conditions_1"
    path = os.path.join(working_dir, directory)

    if os.path.isdir(path):
        i = 2
        while os.path.isdir(path):
            directory = "Conditions_" + str(i)
            path = os.path.join(working_dir, directory)
            i += 1

    os.mkdir(path)

    new_working_dir = path

    new_path = os.path.join(new_working_dir, 'Torch_Data')
    os.mkdir(new_path)
    
    return new_working_dir


# In[4]:


# Establish aberrations

def establish_aberrations(Cs, df):
    """Create a list of aberrations. Initialise with starting values."""

    aberrations = []

    # Krivanek aberration coefficient, Haider aberration coefficient, colloquial name, amplitude, angle, n, m
    # n and m are constants (shouldn't be differentiable), amplitude and angle should be
    aberrations.append(aberration("C10", "C1", "Defocus          ", torch.tensor([df], requires_grad=True), 0.0, 1, 0.0)) # changed float to tensor
    aberrations.append(aberration("C30", "C3", "3rd order spher. ", torch.tensor([Cs], requires_grad = False), 0.0, 3, 0))
    
    return aberrations


# In[5]:


def save_sim_image(sim_image, iterations, defocus, aperture, Cs, working_dir, final_defocus):
    plt.imshow(sim_image.detach().cpu().numpy())
    
    cs_mm = Cs/1e7
    
    plt.title(u'Simulated image of protein\n after {iterations} iterations\n defocus={defocus} Å Cs = {cs} mm\n aperture={aperture} mrad'.format(iterations=iterations, defocus=round(final_defocus, 2), cs=cs_mm, aperture=aperture), fontsize=15)
    
    filepath = working_dir + '/Sim_end_image.jpg'
    filepath_2 = working_dir + '/Torch_Data/image_data.pt'
    
    plt.xticks([])
    plt.yticks([])

    plt.savefig(filepath, dpi=500, bbox_inches="tight")
    plt.show()
    torch.save(sim_image, filepath_2)


# In[6]:


def all_ctf_same_chart(GT_defocus, cutoff, Cs, working_dir, df_start, df_end):
    
    ctf_1 = CTF(energy=eV, semiangle_cutoff=cutoff, Cs=Cs, defocus=GT_defocus)
    x = ctf_1.profiles(max_semiangle=30)["ctf"]

    ctf_2 = CTF(energy=eV, semiangle_cutoff=cutoff, Cs=Cs, defocus=df_start)
    y = ctf_2.profiles(max_semiangle=30)["ctf"]
    
    ctf_3 = CTF(energy=eV, semiangle_cutoff=cutoff, Cs=Cs, defocus=df_end)
    z = ctf_3.profiles(max_semiangle=30)["ctf"]
    
    x_coord = [round(i/500 * 30, 2) for i in range(1,501)] # 30 is from the max semiangle above
    x_coord_k = [k_angstrom * 2 * angle * 1e-3 for angle in x_coord]
    
    plt.plot(x_coord_k, x.array)
    plt.plot(x_coord_k, y.array)
    plt.plot(x_coord_k, z.array)
    
    plt.legend(['Experimental', 'Simulation start', 'Simulation end'])
    
    plt.title('CTF for experimental data, simulation start and simulation end')
    
    plt.ylim([-1.5,1.5])
    plt.xlim([0, 1.3])
    
    plt.ylabel('CTF')
    plt.xlabel(u'k (Å\u207B\u00B9)')
    
    filepath = working_dir + '/all_CTF'
    
    plt.savefig(filepath, dpi=500, bbox_inches="tight")
    plt.clf()


# In[7]:


def save_data(loss_data, ab_data, working_dir, distance_from_GT, aperture):
    
    filepath = working_dir + '/Final_Defocus_curve.jpg'
    filepath_2 = working_dir + '/Loss_curve.jpg'
    
    plt.plot(ab_data)
    plt.title('Final defocus value against number of iterations\n for starting {distance} Å from GT and objective aperture of {aperture} mrad'.format(distance=distance_from_GT, aperture=aperture))
    plt.ylabel("Final defocus value")
    plt.xlabel("Number of iterations")
    plt.savefig(filepath, dpi=500, bbox_inches="tight")
    #plt.show()
    plt.clf()
    
    plt.plot(loss_data)
    plt.title('Loss against number of iterations\n for starting {distance} Å from GT and objective aperture of {aperture} mrad'.format(distance = distance_from_GT, aperture=aperture))
    plt.ylabel("Loss")
    plt.xlabel("Number of iterations")
    plt.savefig(filepath_2, dpi=500, bbox_inches="tight")
    #plt.show()
    plt.clf()
    # Save Data
    
    filename = working_dir + '/CSV_data.csv'
    x = "Iteration, Loss, Defocus"
    
    f = open(filename, "a")
    f.write(x)
    f.write("\n")

    for i in range(0, len(loss_data)):
        n_iter = i+1
        line = str(n_iter) +  ", " + str(loss_data[i]) + ", " + str(ab_data[i])
        f.write(line)
        f.write("\n")

    f.close() 
    


# In[8]:


def log_exp_conditions(learning_rate, loss_modulus, eV, Cs, aperture, distance_from_GT, n_iters, cropped, working_dir):
    """Create a text file which stores the experimental conditions."""
    filepath = working_dir + "/exp_conditions.txt"
    
    file = open(filepath, "w")
    
    e_kV = eV/1000
    
    file.write('Learning Rate: {lr}\nLoss Modulus: {lm}\nEnergy: {e} keV\nCs: {cs} mm\nAperture: {ap} mrad\n{x} away from GT\n{y} iterations\nCropped: {cropped}'.format(lr=learning_rate, lm=loss_modulus, e=e_kV, cs=Cs, ap=aperture, x=distance_from_GT, y=n_iters, cropped=cropped))
    file.close()


# In[9]:


def plot_all_loss(loss_curves, runs):
    plt.clf()
    legends = []
    
    for i in range(len(runs)):
        distance = runs[i][0]
        x = str(distance) + " Å away"
        legends.append(x)
        plt.plot(loss_curves[i])
    
    plt.title("Loss against number of iterations for a series of experimental defocuses")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.legend(legends)

    filepath = main_working_dir + "/All_Loss_Curve"

    plt.savefig(filepath, dpi=500, bbox_inches="tight")
    plt.clf()


# In[10]:


def create_table_data(final_losses, final_defocuses, runs):
    
    filename = main_working_dir + "/table_data.csv"
    GT_df = +500
    
    x = "Distance from GT, Final Loss, Starting defocus, Final defocus value, Absolute error, Percentage error"
    
    f = open(filename, "w")
    f.write(x)
    f.write("\n")

    for i in range(0, len(runs)):
        distance = runs[i][0]
        absolute_error = abs(GT_df-final_defocuses[i])
        percentage_error = (absolute_error / GT_df) * 100
        starting_defocus = GT_df - distance
        
        line = str(distance) +  ", " + str(final_losses[i]) + ", " + str(starting_defocus) + ", "+ str(final_defocuses[i]) + ", " + str(absolute_error) + ", " + str(percentage_error)
        f.write(line)
        f.write("\n")

    f.close()


# In[11]:


def save_ctf(cutoff, Cs, category, working_dir, defocus_value):
    
    ctf = CTF(energy=eV, semiangle_cutoff=cutoff, Cs=Cs, defocus=defocus_value)
    x = ctf.profiles(max_semiangle=30)["ctf"]
    cs_mm = Cs/1e7
    
    if category == "Experimental":
        plot_title = 'CTF for experimental dummy data\n defocus={defocus_value} Å, Cs = {cs} mm\n100 keV, objective aperture = {cutoff} mrad'
        filepath = working_dir + '/CTF_experimental'
    elif category == "Simulation start":
        plot_title = 'CTF for simulation start\n defocus={defocus_value} Å, Cs = {cs} mm\n100 keV, objective aperture = {cutoff} mrad'
        filepath = working_dir + '/CTF_start'
    else:
        plot_title = 'CTF for simulation end\n defocus={defocus_value} Å, Cs = {cs} mm\n100 keV, objective aperture = {cutoff} mrad'
        filepath = working_dir + '/CTF_end'
    
    
    x_coord = [round(i/500 * 30, 2) for i in range(1,501)] # 30 is from the max semiangle above
    x_coord_k = [k_angstrom * 2 * angle * 1e-3 for angle in x_coord]
    
    plt.plot(x_coord_k, x.array)
    
    plt.title(plot_title.format(defocus_value=round(defocus_value, 2), cs=cs_mm, cutoff=cutoff))
    plt.ylim([-1.5,1.5])
    plt.xlim([0, 1.3])
    plt.ylabel('CTF')
    plt.xlabel(u'k (Å\u207B\u00B9)')
    plt.savefig(filepath, dpi=500, bbox_inches="tight")
    plt.clf()
    
    # Save the CTF data
    
    filename = working_dir + '/CTF_data_{x}.csv'.format(x=category)
    first_line = 'k, CTF values'
    
    f = open(filename, "a")
    f.write(first_line)
    f.write("\n")

    for data in x.array:
        line = str(data) + ", " + str(x_coord_k[i])
        f.write(line)
        f.write("\n")
    f.close()
    


# In[12]:


def log_test_information(experimental_conditions, main_working_dir, Cs, aperture):
    filename = main_working_dir + "/tests_run.csv"
    
    x = 'Distance from GT, Number of iterations, Learning Rate, Cs, Aperture'
    
    cs_mm = str(Cs/1e7) # in mm
    aperture = str(aperture)
    
    f = open(filename, "a")
    f.write(x)
    f.write("\n")
    
    for test in experimental_conditions:
        distance = str(test[0])
        iterations = str(test[1])
        lr = str(test[2])

        line = distance + ", " + iterations + ", " + lr + ", " + cs_mm + ", " + aperture 
        f.write(line)
        f.write("\n")
    
    f.close()
        
    


# In[20]:


def noisy(noise_typ, noise_value, image):

    if noise_typ == "gaussian":
        row,col = image.shape
        mean = 0
        var = torch.var(image) * noise_value
        sigma = var**0.5
        gauss = torch.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "poisson":
        vals = len(torch.unique(image))
        vals = 2.1 ** torch.ceil(np.log2(vals))
        noisy = torch.poisson(image * vals) / float(vals)
        return noisy


# In[26]:


def load_experimental_data(noise):
    
    ref_data = torch.load("/home/hremadmin/Documents/Project/ML_files/Protein_Work/Protein_Data_Lowena_2.pt") # added _Lowena
    
    ref_data = ref_data.to(torch.float32)
    
    # apply noise
    
    print(noise)
    
    if noise != None:
        noise_type = "gaussian"
        noise_value = float(noise[1:])

        ref_data = noisy(noise_type, noise_value, ref_data)
    
    plt.clf()
    title = u'Protein reference data\n defocus = +500 Å Cs = 0.5 mm\n aperture = 15 mrad'
    plt.imshow(amplitude(ref_data))
    plt.title(title, fontsize=15)
    plt.xticks([])
    plt.yticks([])

    filepath = working_dir + '/Exp_image.jpg'
    plt.savefig(filepath, dpi=500, bbox_inches="tight")
    plt.show()
    

    return ref_data
    


# In[14]:


# Set up our initial guess for the crystal HRTEM
# It uses the aberrations (including defocus) as set above

def create_initial_guess(aberrations, aperture, initial_df):
    
    df = torch.tensor([initial_df], requires_grad=True)
    
    output = pyms.HRTEM(
        dps_pyms,
        gridshape,
        eV,
        aperture,
        thicknesses,
        subslices=subslices,
        aberrations=aberrations,
        df = df,
        tiling=tiling,
        nfph=1,
        showProgress='notebook',
        dtype = torch.float32,
        apply_ctf = False
    )

    
    return output, df


# In[15]:


def take_centre(input_tensor):
    
    if input_tensor.shape == torch.Size([1, 683, 683]):
        pieces = torch.chunk(input_tensor[0], 3, dim=0)
        pieces_2 = torch.chunk(pieces[1], 3, dim=1)
        pieces_3 = torch.chunk(pieces_2[1], 4, dim=0)
        a = torch.cat((pieces_3[1],pieces_3[2]))
        pieces_4 = torch.chunk(a, 4, dim=1)
        output_tensor = torch.cat((pieces_4[1], pieces_4[2]), dim=1)
        #plt.imshow(abs(output_tensor.detach().cpu().numpy()))
    else:
        pieces = torch.chunk(input_tensor, 3, dim=0)
        pieces_2 = torch.chunk(pieces[1], 3, dim=1)
        pieces_3 = torch.chunk(pieces_2[1], 4, dim=0)
        a = torch.cat((pieces_3[1],pieces_3[2]))
        pieces_4 = torch.chunk(a, 4, dim=1)
        output_tensor = torch.cat((pieces_4[1], pieces_4[2]), dim=1)
        #plt.imshow(abs(output_tensor.detach().cpu().numpy()))
        
    #plt.clf()
    
    return output_tensor


# In[17]:


def optimise2(aberrations, output, initial_df, ref_data, n_iters, learning_rate, loss_multiplier, GT_defocus, aperture, Cs, df, working_dir):
    """Go through and refine the aberrations to minimise difference between initial guess (output) and experimental image"""

    bw_limit_size = size_of_bandwidth_limited_array(gridshape)
    rsize = np.asarray(dps_pyms.unitcell[:2]) * np.asarray(tiling) # will this work?
    
    cs_mm = Cs/1e7
    
    temp_ref = torch.fft.fftn(ref_data)
    temp_ref = torch.fft.fftshift(temp_ref)
    temp_ref = torch.log(torch.abs(temp_ref)**2)
    
    filepath = working_dir + '/ref_FT.jpg'
    plt.imshow(temp_ref.detach().cpu().numpy(), vmax=10)
    plt.title("Fourier Space Ref Data")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filepath, dpi=500, bbox_inches="tight")
    plt.show()
    plt.clf()
    
    
    params = [df]
    
    
    optimizer = torch.optim.Adam(params, lr = learning_rate)


    criterion = torch.nn.MSELoss()

    loss_curve = []
    defocus_curve = []

    start = time.time() 
    
    dtype = torch.float32 
    cdtype = real_to_complex_dtype_torch(dtype)
    
    device = 'cpu'
    
    print("Starting df: ", df.item()) 


    for iter in range(n_iters):
        
        ctf = torch.stack(
                [
                    make_contrast_transfer_function(
                        bw_limit_size, rsize, eV, aperture, df = df, aberrations=aberrations
                    )
                ]
            ).type(cdtype)
        

        hrtem_image = amplitude(torch.fft.ifftn(ctf*output, dim = (-2,-1)))
        sim_image = hrtem_image[0]
        
        hrtem_image = hrtem_image.to(torch.float64)
        amplitude_spectrum = torch.fft.fftshift(torch.fft.fftn(hrtem_image))
        log_amplitude_spectrum = torch.log(torch.abs(amplitude_spectrum)**2)

        
        if iter == 0:
            
            # Save starting image
            plt.imshow(sim_image.detach().cpu().numpy())
            plt.title(u'Simulation start image for protein\n defocus={defocus} Å Cs = {cs} mm\n aperture={aperture} mrad'.format(defocus=initial_df, cs=cs_mm, aperture=aperture), fontsize=15)
            plt.xticks([])
            plt.yticks([])

            filepath = working_dir + '/Sim_start_image.jpg'
            plt.savefig(filepath, dpi=500, bbox_inches="tight")
            plt.show()
            plt.clf()

            # Save starting FT
            filepath = working_dir + '/sim_FT_start.jpg'
            plt.imshow(log_amplitude_spectrum[0].detach().cpu().numpy(), vmax=10)
            plt.title("Fourier space simulated image START")
            plt.xticks([])
            plt.yticks([])
            plt.savefig(filepath, dpi=500, bbox_inches="tight")
            plt.show()
            plt.clf()
            
            
        elif iter == n_iters-1:
            # Save ending FT
            filepath = working_dir + '/sim_FT_end.jpg'
            plt.imshow(log_amplitude_spectrum[0].detach().cpu().numpy(), vmax=10)
            plt.title("Fourier space simulated image END")
            plt.xticks([])
            plt.yticks([])
            plt.savefig(filepath, dpi=500, bbox_inches="tight")
            plt.show()
            plt.clf()
        

        optimizer.zero_grad()
        
        
        # New MSE Loss criterion

        # Uncropped
        loss = criterion(log_amplitude_spectrum[0], temp_ref)
        
        #print("df:", df.item())
        
        
        loss.backward()

        optimizer.step()

        loss_curve.append(loss.item())
        
        defocus_curve.append(df.item())

    end = time.time()
    
    runtime = end-start
    
    save_sim_image(sim_image, n_iters, GT_defocus, aperture, Cs, working_dir, defocus_curve[-1])
    
    return runtime, loss_curve, defocus_curve


# In[22]:


def main(GT_defocus, n_iterations, lr, loss_multiplier, aperture, Cs, distance_from_GT, noise, working_dir):  
    # Load in "experimental" data
    
    ref_data = load_experimental_data(noise)
    ref_data = ref_data.type(torch.complex128)
    
    save_ctf(aperture, Cs, "Experimental", working_dir, GT_defocus)
    
    initial_df = GT_defocus - distance_from_GT
    
    aberrations = establish_aberrations(Cs, initial_df)
    
    output, df = create_initial_guess(aberrations, aperture, initial_df)
    
    save_ctf(aperture, Cs, "Simulation start", working_dir, initial_df)
    
    runtime, loss_data, ab_data = optimise2(aberrations, output, initial_df, ref_data, n_iterations, lr, loss_multiplier, GT_defocus, aperture, Cs, df, working_dir)
    save_ctf(aperture, Cs, "Simulation end", working_dir, ab_data[-1])
    
    all_ctf_same_chart(GT_defocus, aperture, Cs, working_dir, initial_df, ab_data[-1])
    
    return runtime, loss_data, ab_data
    
    


# In[31]:


# Main code


h = 6.62607015e-34
m = 9.11e-31
e = 1.6e-19

wavelength = h / math.sqrt(2*m*eV*e)
wavelength_A = wavelength * 1e10
k_angstrom = 1/wavelength_A

# inputs should be initial guess (-500 GT)

# Distance from GT, number of iterations, learning rate

# For example, this will run 3 tests:
# 1. 400 from GT, 5000 iterations, learning rate of 10, no noise
# 2. 300 from GT, 1000 iterations, learning rate of 10, 5.0 Gaussian noise
# 3. 150 from GT, 2000 iterations, learning rate of 10, 15.0 Gaussian noise


runs = [
    [400.0, 5000, 10, None],
    [300.0, 1000, 10 , "G5.0"],
    [150.0, 2000, 10, "G15.0"]
]


# Constants
loss_multiplier = 1e9

st = time.time()

main_working_dir = create_working_dir()
print(main_working_dir)

aperture = 15
Cs = 0.5e7
GT_defocus = +500.0

log_test_information(runs, main_working_dir, Cs, aperture)

i = 1

loss_curves = []
final_losses = []
final_defocuses = []


for run in runs:
    working_dir = create_folders(main_working_dir)
    distance_from_GT = run[0]
    n_iterations = run[1]
    lr = run[2]
    noise = run[3]
    runtime, loss_data, ab_data = main(GT_defocus, n_iterations, lr, loss_multiplier, aperture, Cs, distance_from_GT, noise, working_dir)
    
    loss_curves.append(loss_data)
    final_losses.append(loss_data[-1])
    final_defocuses.append(ab_data[-1])
    
    save_data(loss_data, ab_data, working_dir, distance_from_GT, aperture)
    
    i += 1
    
    initial_df = GT_defocus - distance_from_GT

    print('Final defocus value for initial defocus of {initial_df}: {final_df}'.format(initial_df=initial_df,
                                                                                        final_df=ab_data[-1]))
    print('Loss: {loss}'.format(loss=loss_data[-1]))

create_table_data(final_losses, final_defocuses, runs)
plot_all_loss(loss_curves, runs)

et = time.time()

time_to_run = et - st

print("Total runtime: ", str(time_to_run))

