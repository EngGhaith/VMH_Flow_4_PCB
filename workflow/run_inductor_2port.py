
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules')))

import modules.util_stackup_reader as stackup_reader
import modules.util_gds_reader as gds_reader
import modules.util_utilities as utilities
import modules.util_simulation_setup as simulation_setup
import modules.util_meshlines as util_meshlines

import os
from pylab import *
from CSXCAD import ContinuousStructure
from CSXCAD import AppCSXCAD_BIN
from openEMS import openEMS
from openEMS.physical_constants import *

# Model comments
# Simple inductor model, 2 port model with two via ports to substrate, full two port simulation


# ======================== workflow settings ================================

# preview model/mesh only?
# postprocess existing data without re-running simulation?
preview_only = True    
postprocess_only = False

# ===================== input files and path settings =======================

gds_filename = "L_2n0_twoport.gds"      # geometries
XML_filename = "SG13G2.xml"               # stackup

# preprocess GDSII for safe handling of cutouts/holes?
preprocess_gds = False

# merge via polygons with distance less than 1.5 um
merge_polygon_size = 1.5

# get path for this simulation file
script_path = utilities.get_script_path(__file__)

# use script filename as model basename
model_basename = utilities.get_basename(__file__)

# set and create directory for simulation output
sim_path = utilities.create_sim_path (script_path,model_basename)
print('Simulation data directory: ', sim_path)


# ======================== simulation settings ================================

unit   = 1e-6   # geometry is in microns
margin = 200    # distance in microns from GDSII geometry boundary to simulation boundary 

fstart = 0
fstop  = 30e9
numfreq = 401

refined_cellsize = 1.0  # mesh cell size in conductor region

# choices for boundary: 
# 'PEC' : perfect electric conductor (default)
# 'PMC' : perfect magnetic conductor, useful for symmetries
# 'MUR' : simple MUR absorbing boundary conditions
# 'PML_8' : PML absorbing boundary conditions
Boundaries = ['PEC', 'PEC', 'PEC', 'PEC', 'PEC', 'PEC']

cells_per_wavelength = 20   # how many mesh cells per wavelength, must be 10 or more
energy_limit = -50          # end criteria for residual energy (dB)

# ports from GDSII Data, polygon geometry from specified special layer
# note that for multiport simulation, excitations are switched on/off in simulation_setup.createSimulation below
simulation_ports = simulation_setup.all_simulation_ports()
# instead of in-plane port specified with target_layername, we here use via port specified with from_layername and to_layername
simulation_ports.add_port(simulation_setup.simulation_port(portnumber=1, voltage=1, port_Z0=50, source_layernum=201, from_layername='SUBGND', to_layername='TopMetal1', direction='z'))
simulation_ports.add_port(simulation_setup.simulation_port(portnumber=2, voltage=1, port_Z0=50, source_layernum=202, from_layername='SUBGND', to_layername='TopMetal1', direction='z'))

# ======================== simulation ================================

# get technology stackup data
materials_list, dielectrics_list, metals_list = stackup_reader.read_substrate (XML_filename)
# get list of layers from technology
layernumbers = metals_list.getlayernumbers()
layernumbers.extend(simulation_ports.portlayers)

# read geometries from GDSII, only purpose 0
allpolygons = gds_reader.read_gds(gds_filename, layernumbers, purposelist=[0], metals_list=metals_list, preprocess=preprocess_gds, merge_polygon_size=merge_polygon_size)


# calculate maximum cellsize from wavelength in diecletric
wavelength_air = 3e8/fstop
max_cellsize = (wavelength_air/unit)/(sqrt(materials_list.eps_max)*cells_per_wavelength) 


########### create model, run and post-process ###########


# Create simulation for port 1 and 2 excitation, return value is list of data paths, one for each excitation
data_paths = []
for excite_ports in [[1],[2]]:  # list of ports that are excited one after another
    FDTD = openEMS(EndCriteria=exp(energy_limit/10 * log(10)))
    FDTD.SetGaussExcite( (fstart+fstop)/2, (fstop-fstart)/2 )
    FDTD.SetBoundaryCond( Boundaries )
    FDTD = simulation_setup.setupSimulation (excite_ports, simulation_ports, FDTD, materials_list, dielectrics_list, metals_list, allpolygons, max_cellsize, refined_cellsize, margin, unit, xy_mesh_function=util_meshlines.create_xy_mesh_from_polygons)
    data_paths.append(simulation_setup.runSimulation (excite_ports, FDTD, sim_path, model_basename, preview_only, postprocess_only))


if preview_only==False:

    print('Begin data evaluation')

    # define dB function for S-parameters
    def dB(value):
        return 20.0*np.log10(np.abs(value))        

    # define phase function for S-parameters
    def phase(value):
        return angle(value, deg=True) 

    f = np.linspace(fstart,fstop,numfreq)

    # get results, CSX port definition is read from simulation ports object
    # S12, S22 is available because we have simulated both port1 and port2  excitation
    s11 = utilities.calculate_Sij (1, 1, f, sim_path, simulation_ports)
    s21 = utilities.calculate_Sij (2, 1, f, sim_path, simulation_ports)
    s12 = utilities.calculate_Sij (1, 2, f, sim_path, simulation_ports)
    s22 = utilities.calculate_Sij (2, 2, f, sim_path, simulation_ports)

    s2p_name = os.path.join(sim_path, model_basename + '.s2p')
    utilities.write_snp (np.array([[s11, s21],[s12,s22]]),f, s2p_name)

    # calculate inductor parameters using Z parameters
    z11 = utilities.calculate_Zij_2port (1, 1, f, sim_path, simulation_ports)
    z21 = utilities.calculate_Zij_2port (2, 1, f, sim_path, simulation_ports)
    z12 = utilities.calculate_Zij_2port (1, 2, f, sim_path, simulation_ports)
    z22 = utilities.calculate_Zij_2port (2, 2, f, sim_path, simulation_ports)

    # ignore divide by zero warning during inductor calculation at DC
    np.seterr(divide='ignore', invalid='ignore')
    
    zdiff = z11-z12-z21+z22
    omega = 2*np.pi*f
    Qdiff = zdiff.imag/zdiff.real
    Ldiff = zdiff.imag/omega
    Rdiff = zdiff.real

    # print some inductor data
    # get series L and series R at frequency of interest
    targetfreq = 10e9
    findex = where (f>=targetfreq)[0]
    findex = findex.item(0)

    print('\nDifferential inductor parameters')
    
    print(f"Frequency [GHz]: {f[findex]/1e9:.3f}")  
    print(f"Series L  [nH] : {Ldiff[findex]*1e9:.3f}")  
    print(f"Series R  [Ohm]: {Rdiff[findex]:.3f}") 
    print(f"Q factor       : {Qdiff[findex]:.2f}")  
    print('----------------------')
    print(f"L_DC      [nH] : {Ldiff[1]*1e9:.3f}") 
    print(f"R_DC      [Ohm]: {Rdiff[0]:.3f}")  
    print(f"Peak Q         : {max(Qdiff):.2f}") 

    print('\nStarting plots')

    figure()
    plot(f/1e9, dB(s11), 'k-',  linewidth=2, label='S11 [dB]')
    plot(f/1e9, dB(s22), 'r--',  linewidth=2, label='S22 [dB]')
    grid()
    legend()
    xlabel('Frequency (GHz)')

    figure()
    plot(f/1e9, dB(s21), 'k-',  linewidth=2, label='S21 [dB]')
    plot(f/1e9, dB(s12), 'r--',  linewidth=2, label='S12 [dB]')
    grid()
    legend()
    xlabel('Frequency (GHz)')


    figure()
    plot(f/1e9, Ldiff*1e9, 'k-', linewidth=2, label='Lseries [nH]')
    ylim(0, 10)
    xlim(0, fstop/1e9)
    grid()
    legend()
    xlabel('Frequency (GHz)')
    
    figure()
    plot(f/1e9, Qdiff, 'k-', linewidth=2, label='Q factor')
    ylim(0, 30)
    xlim(0, fstop/1e9)
    grid()
    legend()
    xlabel('Frequency (GHz)')


    # Show all plots
    show()

