#!/usr/bin/env python
# coding: utf-8

# # Crystalline FT Optimisation

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


crystal = pyms.structure.fromfile(
    "/home/hremadmin/py_multislice_differentiable/py_multislice_differentiable_tem/Demos/Structures/SrTiO3_CeO2_interface.xyz", atomic_coordinates="cartesian"
)

# A few maniupulations to remove vaccuum at edges and create a psuedo-periodic structure
crystal = crystal.resize([0.1, 0.76], axis=0)

other = deepcopy(crystal).resize([0.017, 0.99], axis=0)
other.reflect([0])
crystal = crystal.concatenate(other, axis=0)

# Subslicing of crystal for multislice
subslices = [0.33, 0.66, 1.0]

# Grid size in pixels
gridshape = [1024, 1024]

# Tile structure in multislice for a square grid
tiling = [1, 7]

# Probe accelerating voltage in eV
eV = 3e5


# Set up series of thicknesses
thicknesses = np.array([100])


# In[76]:


def create_working_dir():
    directory = "Run_1"
    parent_dir = '/home/hremadmin/Documents/Project/ML_files/FT_crystalline/'
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


def create_folders(working_dir, defocus_values):
    
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
    
    # Make a directory for each defocus value being tested
    folder_names = ['Torch_Data'] + [str(x)+"_data" for x in defocus_values]

    for name in folder_names:
        new_path = os.path.join(new_working_dir, name)
        os.mkdir(new_path)
    
    return new_working_dir


# In[4]:


# Establish aberrations

def establish_aberrations(Cs):
    """Create a list of aberrations. Initialise with starting values."""

    aberrations = []

    # Krivanek aberration coefficient, Haider aberration coefficient, colloquial name, amplitude, angle, n, m
    # n and m are constants (shouldn't be differentiable), amplitude and angle should be

    aberrations.append(aberration("C30", "C3", "3rd order spher. ", torch.tensor([Cs], requires_grad = False), 0.0, 3, 0))
    
    return aberrations


# In[25]:


def save_sim_image(sim_image, iterations, defocus, aperture, Cs, working_dir, final_defocus):
    plt.imshow(sim_image.detach().cpu().numpy())
    
    cs_mm = Cs/1e7
    
    plt.title(u'Simulated image of SrTiO\u2083/CeO\u2082 interface\n after {iterations} iterations\n defocus={defocus} Å 1 frozen phonon pass Cs = {cs} mm\n aperture={aperture} mrad'.format(iterations=iterations, defocus=round(final_defocus, 2), cs=cs_mm, aperture=aperture), fontsize=15)
    
    filepath = working_dir + '/{defocus}_data/Sim_end_image_{defocus}_{aperture}.jpg'.format(defocus=defocus, aperture=aperture)
    filepath_2 = working_dir + "/Torch_Data/image_data_"+ str(defocus) + "_" + str(aperture) + "_" + str(iterations) + ".pt"
    
    plt.xticks([])
    plt.yticks([])

    plt.savefig(filepath, dpi=500, bbox_inches="tight")
    plt.show()
    torch.save(sim_image, filepath_2)


# In[6]:


def generate_experimental_data(GT_defocus, aperture, Cs, noise, working_dir):
    """Generate different starting experimental 'dummy' images"""
    
    df = torch.tensor([GT_defocus])
    cs_mm = Cs/1e7
    
    aberrations = establish_aberrations(Cs)
    
    
    output = pyms.HRTEM(
        crystal,
        gridshape,
        eV,
        aperture,
        thicknesses,
        subslices=subslices,
        aberrations=aberrations,
        df=df,
        tiling=tiling,
        nfph=1,
        showProgress='notebook',
        dtype = torch.float32,
    )
    
    
    
    if noise != None: # If we want to add noise, split the noise code
        if noise[0] == "P":
            noise_type = "poisson"
            noise_value = None
            title = u'Dummy image for SrTiO\u2083/CeO\u2082 interface\n defocus={defocus} Å 1 frozen phonon pass Cs = {cs} mm\n aperture={aperture} mrad with Poisson noise'.format(defocus=GT_defocus, cs = cs_mm, aperture=aperture)
        else:
            noise_type = "gaussian"
            noise_value = float(noise[1:])
            title = u'Dummy image for SrTiO\u2083/CeO\u2082 interface\n defocus={defocus} Å 1 frozen phonon pass Cs = {cs} mm\n aperture={aperture} mrad with {noise} Gaussian noise'.format(defocus=GT_defocus, cs = cs_mm, aperture=aperture, noise=noise_value)
    
        output = noisy(noise_type, noise_value, output)
    else:
        title = u'Dummy image for SrTiO\u2083/CeO\u2082 interface\n defocus={defocus} Å 1 frozen phonon pass Cs = {cs} mm\n aperture={aperture} mrad'.format(defocus=GT_defocus, cs = cs_mm, aperture=aperture)
       
    plt.imshow(amplitude(output), vmin=0)
    
    plt.title(title, fontsize=15)
    plt.xticks([])
    plt.yticks([])

    filepath = working_dir + '/{defocus}_data/Exp_image_{defocus}_{aperture}.jpg'.format(defocus=GT_defocus, aperture=aperture)
    plt.savefig(filepath, dpi=500, bbox_inches="tight")
    plt.show()
    
    output = output.type(torch.float64)
    
    return output


# In[7]:


def noisy(noise_typ, noise_value, image):
    if noise_typ == "gaussian":
        row,col = image.shape
        mean = 0
        var = np.var(image) * noise_value
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2.1 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy


# In[8]:


def all_ctf_same_chart(defocus, cutoff, Cs, working_dir, df_start, df_end):
    
    ctf_1 = CTF(energy=eV, semiangle_cutoff=cutoff, Cs=Cs, defocus=defocus)
    x = ctf_1.profiles(max_semiangle=15)["ctf"]

    ctf_2 = CTF(energy=eV, semiangle_cutoff=cutoff, Cs=Cs, defocus=df_start)
    y = ctf_2.profiles(max_semiangle=15)["ctf"]
    
    ctf_3 = CTF(energy=eV, semiangle_cutoff=cutoff, Cs=Cs, defocus=df_end)
    z = ctf_3.profiles(max_semiangle=15)["ctf"]
    
    x_coord = [round(i/500 * 15, 2) for i in range(1,501)] # 20 is from the max semiangle above
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
    
    filepath = working_dir + '/{defocus}_data/all_CTF'.format(defocus=defocus)
    
    plt.savefig(filepath, dpi=500, bbox_inches="tight")
    plt.clf()


# In[9]:


def save_data(loss_data, ab_data, working_dir, defocus, aperture):
    
    folder_name = "/" + str(defocus) + "_data"
    
    filepath = working_dir + folder_name + '/Final_Defocus_curve_{defocus}_{aperture}.jpg'.format(defocus=defocus, aperture=aperture)
    filepath_2 = working_dir + folder_name + '/Loss_curve_{defocus}_{aperture}.jpg'.format(defocus=defocus, aperture=aperture)
    
    plt.plot(ab_data)
    plt.title('Final defocus value against number of iterations\n for GT defocus of {defocus} Å and objective aperture of {aperture} mrad'.format(defocus=defocus, aperture=aperture))
    plt.ylabel("Final defocus value")
    plt.xlabel("Number of iterations")
    plt.savefig(filepath, dpi=500, bbox_inches="tight")
    #plt.show()
    plt.clf()
    
    plt.plot(loss_data)
    plt.title('Loss against number of iterations\n for GT defocus of {defocus} Å and objective aperture of {aperture} mrad'.format(defocus=defocus, aperture=aperture))
    plt.ylabel("Loss")
    plt.xlabel("Number of iterations")
    plt.savefig(filepath_2, dpi=500, bbox_inches="tight")
    #plt.show()
    
    # Save Data
    
    filename = working_dir + folder_name + '/CVS_data_DF_{defocus}_APP_{aperture}.csv'.format(defocus=defocus, aperture=aperture)
    x = "Iteration, Loss, Final defocus value"
    
    f = open(filename, "a")
    f.write(x)
    f.write("\n")

    for i in range(0, len(loss_data)):
        n_iter = i+1
        line = str(n_iter) +  ", " + str(loss_data[i]) + ", " + str(ab_data[i])
        f.write(line)
        f.write("\n")

    f.close() 
    


# In[63]:


def log_exp_conditions(learning_rate, loss_modulus, eV, Cs, aperture, distance_from_GT, n_iters, cropped, working_dir):
    """Create a text file which stores the experimental conditions."""
    filepath = working_dir + "/exp_conditions.txt"
    
    file = open(filepath, "w")
    
    e_kV = eV/1000
    
    file.write('Learning Rate: {lr}\nLoss Modulus: {lm}\nEnergy: {e} keV\nCs: {cs} mm\nAperture: {ap} mrad\n{x} away from GT\n{y} iterations\nCropped: {cropped}'.format(lr=learning_rate, lm=loss_modulus, e=e_kV, cs=Cs, ap=aperture, x=distance_from_GT, y=n_iters, cropped=cropped))
    file.close()


# In[11]:


def create_table_data(final_losses, final_defocuses, defocus_values, working_dir):
    
    filename = working_dir + "/table_data.csv"
    
    x = "Defocus, Loss, Final defocus value, Absolute error, Percentage error"
    
    f = open(filename, "a")
    f.write(x)
    f.write("\n")

    for i in range(0, len(final_losses)):
        df = defocus_values[i]
        absolute_error = abs(df-final_defocuses[i])
        percentage_error = (absolute_error / df) * 100
        
        line = str(df) +  ", " + str(final_losses[i]) + ", " + str(final_defocuses[i]) + ", " + str(absolute_error) + ", " + str(percentage_error)
        f.write(line)
        f.write("\n")

    f.close()


# In[12]:


def plot_all_loss(loss_curves, working_dir):
    plt.clf()
    plt.plot(loss_curves[0])
    plt.plot(loss_curves[1])
    plt.plot(loss_curves[2])
    plt.plot(loss_curves[3])
    plt.legend(['No oscillations', 'One oscillation', 'Few oscillations', 'Many oscillations'])
    plt.title("Loss against number of iterations for a series of experimental defocuses")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")

    filepath = working_dir + "/All_Loss_Curve"

    plt.savefig(filepath, dpi=500, bbox_inches="tight")
    plt.clf()


# In[13]:


def save_ctf(defocus, cutoff, Cs, category, working_dir, defocus_value):
    
    ctf = CTF(energy=eV, semiangle_cutoff=cutoff, Cs=Cs, defocus=defocus_value)
    x = ctf.profiles(max_semiangle=15)["ctf"]
    cs_mm = Cs/1e7
    
    if category == "Experimental":
        plot_title = 'CTF for experimental dummy data\n defocus={defocus_value} Å, Cs = {cs} mm\n300 keV, objective aperture = {cutoff} mrad'
        filepath = working_dir + '/{defocus}_data/CTF_experimental'.format(defocus=defocus)
    elif category == "Simulation start":
        plot_title = 'CTF for simulation start\n defocus={defocus_value} Å, Cs = {cs} mm\n300 keV, objective aperture = {cutoff} mrad'
        filepath = working_dir + '/{defocus}_data/CTF_start'.format(defocus=defocus)
    else:
        plot_title = 'CTF for simulation end\n defocus={defocus_value} Å, Cs = {cs} mm\n300 keV, objective aperture = {cutoff} mrad'
        filepath = working_dir + '/{defocus}_data/CTF_end'.format(defocus=defocus)
    
    
    x_coord = [round(i/500 * 15, 2) for i in range(1,501)] # 20 is from the max semiangle above
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
    


# In[14]:


def log_test_information(experimental_conditions, main_working_dir):
    filename = main_working_dir + "/tests_run.csv"
    
    x = 'A, B, C, D, Aperture (mrad), Cs (mm), Distance from GT, Noise, Number of Iterations, Learning rate'
    
    f = open(filename, "a")
    f.write(x)
    f.write("\n")
    
    for test in experimental_conditions:
        A = str(test[0][0])
        B = str(test[0][1])
        C = str(test[0][2])
        D = str(test[0][3])
        aperture = str(test[1])
        Cs = str(test[2]/1e7)
        distance = str(test[3])
        noise = str(test[4])
        iterations = str(test[5])
        lr = str(test[6])
        
        line = A + ", " + B + ", " + C + ", " + D + ", " + aperture + ", " + Cs + ", " + distance + ", " + noise + ", " + iterations + ", " + lr
        f.write(line)
        f.write("\n")
    
    f.close()
        
    


# In[15]:


# Set up our initial guess for the crystal HRTEM
# It uses the aberrations (including defocus) as set above

def create_initial_guess(aberrations, aperture, initial_df):
    
    df = torch.tensor([initial_df], requires_grad=True)
    
    output = pyms.HRTEM(
        crystal,
        gridshape,
        eV,
        aperture,
        thicknesses,
        subslices=subslices,
        aberrations=aberrations,
        df= df,
        tiling=tiling,
        nfph=1,
        showProgress='notebook',
        dtype = torch.float32,
        apply_ctf = False
    )

    
    return output, df


# In[64]:


def main(n_iters, learning_rate, loss_multiplier, GT_defocus, aperture, Cs, distance_from_GT, noise, cropped, working_dir):
    
    # Load in "experimental" data
    
    ref_data = generate_experimental_data(GT_defocus, aperture, Cs, noise, working_dir)
    ref_data = ref_data.type(torch.complex128)
    
    
    save_ctf(GT_defocus, aperture, Cs, "Experimental", working_dir, GT_defocus)
    
    aberrations = establish_aberrations(Cs)
    
    initial_df = GT_defocus - distance_from_GT
    
    output, df = create_initial_guess(aberrations, aperture, initial_df)
    save_ctf(GT_defocus, aperture, Cs, "Simulation start", working_dir, initial_df)
    
    runtime, loss_data, ab_data = optimise2(aberrations, output, initial_df, ref_data, n_iters, learning_rate, loss_multiplier, GT_defocus, aperture, Cs, df, cropped, working_dir)
    save_ctf(GT_defocus, aperture, Cs, "Simulation end", working_dir, ab_data[-1])
    
    all_ctf_same_chart(GT_defocus, aperture, Cs, working_dir, initial_df, ab_data[-1])
    
    return runtime, loss_data, ab_data
    
    


# In[60]:


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


# In[69]:


def optimise2(aberrations, output, initial_df, ref_data, n_iters, learning_rate, loss_multiplier, GT_defocus, aperture, Cs, df, cropped, working_dir):
    """Go through and refine the aberrations to minimise difference between initial guess (output) and experimental image"""

    bw_limit_size = size_of_bandwidth_limited_array(gridshape)
    rsize = np.asarray(crystal.unitcell[:2]) * np.asarray(tiling)
    
    cs_mm = Cs/1e7
    
    temp_ref = torch.fft.fftn(ref_data)
    temp_ref = torch.fft.fftshift(temp_ref)
    temp_ref = torch.log(temp_ref**2)
    
    filepath = working_dir + "/"+ str(GT_defocus)+"_data" + "/ref_FT.jpg"
    plt.imshow(temp_ref.detach().cpu().numpy(), vmax=10)
    plt.title("Fourier Space Ref Data")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filepath, dpi=500, bbox_inches="tight")
    plt.show()
    plt.clf()
    


    print(df)
    params = [df]
    
    

    optimizer = torch.optim.Adam(params, lr = learning_rate)

    criterion = torch.nn.MSELoss()

    loss_curve = []
    defocus_curve = []

    start = time.time() 
    
    dtype = torch.float32 #torch.float64
    cdtype = real_to_complex_dtype_torch(dtype)
    
    device = 'cpu'


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
        log_amplitude_spectrum = torch.log(amplitude_spectrum**2)
        #plt.imshow(amplitude(log_amplitude_spectrum[0]).detach().cpu().numpy())

        
        
        if iter == 0:
            
            # Save starting image
            plt.imshow(sim_image.detach().cpu().numpy())
            plt.title(u'Simulation start image for SrTiO\u2083/CeO\u2082 interface\n defocus={defocus} Å 1 frozen phonon pass Cs = {cs} mm\n aperture={aperture} mrad'.format(defocus=initial_df, cs=cs_mm, aperture=aperture), fontsize=15)
            plt.xticks([])
            plt.yticks([])

            filepath = working_dir + '/{defocus}_data/Sim_start_image_{df}_{aperture}.jpg'.format(defocus=GT_defocus, df=initial_df, aperture=aperture)
            plt.savefig(filepath, dpi=500, bbox_inches="tight")
            plt.show()
            plt.clf()

            # Save starting FT
            filepath = working_dir + "/"+ str(GT_defocus) + "_data" + "/sim_FT_start.jpg"
            plt.imshow(log_amplitude_spectrum[0].detach().cpu().numpy(), vmax=10)
            plt.title("Fourier space simulated image START")
            plt.xticks([])
            plt.yticks([])
            plt.savefig(filepath, dpi=500, bbox_inches="tight")
            plt.show()
            plt.clf()
            
            
        elif iter == n_iters-1:
            # Save ending FT
            filepath = working_dir + "/"+ str(GT_defocus) + "_data" + "/sim_FT_end.jpg"
            plt.imshow(log_amplitude_spectrum[0].detach().cpu().numpy(), vmax=10)
            plt.title("Fourier space simulated image END")
            plt.xticks([])
            plt.yticks([])
            plt.savefig(filepath, dpi=500, bbox_inches="tight")
            plt.show()
            plt.clf()
        

        optimizer.zero_grad()
        
        
        # New MSE Loss criterion

        
        if cropped == True:
            # Crop to just take the centre
            input_cropped = take_centre(log_amplitude_spectrum)
            ref_cropped = take_centre(temp_ref)
            loss = criterion(amplitude(input_cropped), amplitude(ref_cropped))
        else:
            # Or not cropped
            loss = criterion(amplitude(log_amplitude_spectrum), amplitude(temp_ref))
        
        
        loss.backward()
        

        optimizer.step()

        loss_curve.append(loss.item())
        
        defocus_curve.append(df.item())
        

    end = time.time()
    
    runtime = end-start
    
    save_sim_image(sim_image, n_iters, GT_defocus, aperture, Cs, working_dir, defocus_curve[-1])
    
    return runtime, loss_curve, defocus_curve


# In[73]:


# Main code

# First is defocus values
# Second is aperture
# Third is Cs
# Fourth is distance away from GT
# Fifth is noise code
# Sixth is number of iterations
# Seventh is learning rate
# Eighteth is cropped or not

# This will run one test for Standard conditions (cases A-D), 150 Å from the GT, no noise, 5000 iterations, 
# learning rate of 10 and will include the whole image (will not just crop the centre).

experimental_conditions = [
    [[-75.0, -425.0, -1475.0, -4975.0], 7.5, 1e7, 150.0, None, 5000, 1e1, False],  
]


h = 6.62607015e-34
m = 9.11e-31
e = 1.6e-19

wavelength = h / math.sqrt(2*m*eV*e)
wavelength_A = wavelength * 1e10
k_angstrom = 1/wavelength_A

# Constants
loss_multiplier = 1e9

st = time.time()

main_working_dir = create_working_dir()
print(main_working_dir)

log_test_information(experimental_conditions, main_working_dir)


for i in range(len(experimental_conditions)):
    
    runtimes = []
    ab_curves = []
    loss_curves = []
    final_losses = []
    final_defocuses = []
    
    Cs = experimental_conditions[i][2]
    aperture = experimental_conditions[i][1]
    distance_from_GT = experimental_conditions[i][3]
    noise = experimental_conditions[i][4]
    defocus_values = experimental_conditions[i][0]
    n_iterations = experimental_conditions[i][5]
    learning_rate = experimental_conditions[i][6]
    cropped = experimental_conditions[i][7]
    working_dir = create_folders(main_working_dir, defocus_values)
    print(working_dir)
    
    log_exp_conditions(learning_rate, loss_multiplier, eV, Cs/1e7, aperture, distance_from_GT, n_iterations, cropped, working_dir)
    
    for df in defocus_values:
        runtime, loss_data, ab_data = main(n_iterations, learning_rate, loss_multiplier, df, aperture, Cs, distance_from_GT, noise, cropped, working_dir)
        runtimes.append(runtime)
        ab_curves.append(ab_data)
        loss_curves.append(loss_data)
        save_data(loss_data, ab_data, working_dir, df, aperture)

        print('Final defocus value for initial defocus of {initial_df}: {final_df}'.format(initial_df=df,
                                                                                        final_df=ab_data[-1]))
        print('Loss: {loss}'.format(loss=loss_data[-1]))
        
        final_losses.append(loss_data[-1])
        final_defocuses.append(ab_data[-1])
        
    
    create_table_data(final_losses, final_defocuses, defocus_values, working_dir)
    plot_all_loss(loss_curves, working_dir)

et = time.time()

time_to_run = et - st

print("Total runtime: ", str(time_to_run))

