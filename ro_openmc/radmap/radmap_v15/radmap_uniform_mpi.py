import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import time, timeit
from datetime import datetime
import openmc

###=================Input parameter======================
rad_source_x= [50, 7]
###=======================================

def gen_materials_geometry_tallies(panel_density, e_filter, *energy):
    panel = openmc.Material(name='CdZnTe')
    panel.set_density('g/cm3', panel_density)
    panel.add_nuclide('Cd114', 33, percent_type='ao')
    panel.add_nuclide('Zn64', 33, percent_type='ao')
    panel.add_nuclide('Te130', 33, percent_type='ao')

    insulator = openmc.Material(name='attn')
    insulator.set_density('g/cm3', 1)
    insulator.add_nuclide('Pb208', 11.35)
    
    outer = openmc.Material(name='Outer_CdZnTe')
    outer.set_density('g/cm3', panel_density)
    outer.add_nuclide('Cd114', 33, percent_type='ao')
    outer.add_nuclide('Zn64', 33, percent_type='ao')
    outer.add_nuclide('Te130', 33, percent_type='ao')

    materials = openmc.Materials(materials=[panel, insulator, outer])
    #materials.cross_sections = '/home/rokabe/data1/openmc/endfb71_hdf5/cross_sections.xml'
    materials.export_to_xml()

    os.system("cat materials.xml")

    min_x = openmc.XPlane(x0=-100000, boundary_type='transmission')
    max_x = openmc.XPlane(x0=+100000, boundary_type='transmission')
    min_y = openmc.YPlane(y0=-100000, boundary_type='transmission')
    max_y = openmc.YPlane(y0=+100000, boundary_type='transmission')

    #for S1 layer
    min_x1 = openmc.XPlane(x0=-0.4, boundary_type='transmission')   #!20220124
    max_x1 = openmc.XPlane(x0=+0.4, boundary_type='transmission')
    min_y1 = openmc.YPlane(y0=-0.4, boundary_type='transmission')
    max_y1 = openmc.YPlane(y0=+0.4, boundary_type='transmission')

    #for S2 layer
    min_x2 = openmc.XPlane(x0=-0.5, boundary_type='transmission')
    max_x2 = openmc.XPlane(x0=+0.5, boundary_type='transmission')
    min_y2 = openmc.YPlane(y0=-0.5, boundary_type='transmission')
    max_y2 = openmc.YPlane(y0=+0.5, boundary_type='transmission')

    #for S3 layer
    min_x3 = openmc.XPlane(x0=-5, boundary_type='transmission')
    max_x3 = openmc.XPlane(x0=+5, boundary_type='transmission')
    min_y3 = openmc.YPlane(y0=-5, boundary_type='transmission')
    max_y3 = openmc.YPlane(y0=+5, boundary_type='transmission')
    
    min_xx = openmc.XPlane(x0=-100100, boundary_type='vacuum')
    max_xx = openmc.XPlane(x0=+100100, boundary_type='vacuum')
    min_yy = openmc.YPlane(y0=-100100, boundary_type='vacuum')
    max_yy = openmc.YPlane(y0=+100100, boundary_type='vacuum')

    #s1 region
    s1_region = +min_x1 & -max_x1 & +min_y1 & -max_y1

    #s2 region
    s2_region = +min_x2 & -max_x2 & +min_y2 & -max_y2

    #s3 region
    s3_region = +min_x3 & -max_x3 & +min_y3 & -max_y3

    #s4 region
    s4_region = +min_x & -max_x & +min_y & -max_y
    
    #s5 region
    s5_region = +min_xx & -max_xx & +min_yy & -max_yy

    #define s1 cell
    s1_cell = openmc.Cell(name='s1 cell', fill=panel, region=s1_region)

    #define s2 cell
    s2_cell = openmc.Cell(name='s2 cell', fill=insulator, region= ~s1_region & s2_region)

    # Create a Universe to encapsulate a fuel pin
    cell_universe = openmc.Universe(name='universe', cells=[s1_cell, s2_cell])   #!20220117

    # Create fuel assembly Lattice
    assembly = openmc.RectLattice(name='detector arrays')
    assembly.pitch = (1, 1)
    assembly.lower_left = [-1 * 10 / 2.0] * 2
    assembly.universes = [[cell_universe] * 10] * 10

    # Create root Cell
    arrays_cell = openmc.Cell(name='arrays cell', fill=assembly, region = s3_region)
    root_cell = openmc.Cell(name='root cell', fill=None, region = ~s3_region & s4_region)
    outer_cell = openmc.Cell(name='outer cell', fill=None, region = ~s4_region & s5_region)

    root_universe = openmc.Universe(name='root universe')
    root_universe.add_cell(arrays_cell)
    root_universe.add_cell(root_cell)
    root_universe.add_cell(outer_cell)

    root_universe.plot(width=(22, 22), basis='xy')

    # Create Geometry and export to "geometry.xml"
    geometry = openmc.Geometry(root_universe)
    geometry.export_to_xml()

    os.system("cat geometry.xml")

    # Instantiate an empty Tallies object
    tallies = openmc.Tallies()

    # Instantiate a tally Mesh
    mesh = openmc.RegularMesh(mesh_id=1)
    mesh.dimension = [10, 10]
    mesh.lower_left = [-5, -5]
    mesh.width = [1, 1]

    # Instantiate tally Filter
    mesh_filter = openmc.MeshFilter(mesh)

    # Instantiate the Tally
    tally = openmc.Tally(name='mesh tally')
    
    if e_filter:
        energy_filter = openmc.EnergyFilter(*energy)
        tally.filters = [mesh_filter, energy_filter]
    
    else:
        tally.filters = [mesh_filter]

    tally.scores = ["absorption"]

    # Add mesh and Tally to Tallies
    tallies.append(tally)

    # Instantiate tally Filter
    cell_filter = openmc.CellFilter(s1_cell)

    # Instantiate the tally
    tally = openmc.Tally(name='cell tally')
    tally.filters = [cell_filter]
    tally.scores = ['absorption']
    tally.nuclides = ['Cd114', 'Te130', 'Zn64']

    # Instantiate tally Filter
    distribcell_filter = openmc.DistribcellFilter(s2_cell)

    # Instantiate tally Trigger for kicks
    trigger = openmc.Trigger(trigger_type='std_dev', threshold=5e-5)
    trigger.scores = ['absorption']

    # Instantiate the Tally
    tally = openmc.Tally(name='distribcell tally')
    tally.filters = [distribcell_filter]
    tally.scores = ['absorption']
    tally.nuclides = ['Cd114', 'Te130', 'Zn64']
    tally.triggers = [trigger]

    # Export to "tallies.xml"
    tallies.export_to_xml()

    os.system("cat tallies.xml")

    # Remove old HDF5 (summary, statepoint) files
    os.system('rm statepoint.*')
    os.system('rm summary.*')


#def gen_settings(rad_source1=rad_source_x):
def gen_settings(src_energy=None, src_strength=1, en_a=0, en_b=1, num_particles=10000, batch_size=100, source_x=rad_source_x[0], source_y=rad_source_x[1]): #!20220224
    # Create a point source
    point1 = openmc.stats.Point((source_x, source_y, 0))
    source1 = openmc.Source(space=point1, particle='photon', energy=src_energy, strength=src_strength)  #!20220204    #!20220118
    #point2 = openmc.stats.Point((-50, 6, 0))
    #source2 = openmc.Source(space=point2, particle='photon')
    #point3 = openmc.stats.Point((1, -20, 0))
    #source3 = openmc.Source(space=point3, particle='photon')

    #!====================
    source1.energy = openmc.stats.Uniform(a=en_a, b=en_b)
    #!====================

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.photon_transport = True
    settings.source = [source1] #, source2, source3]     #!20220118
    settings.batches = batch_size
    settings.inactive = 10
    settings.particles = num_particles

    settings.export_to_xml()

    os.system("cat settings.xml")


def run_openmc():
    # Run OpenMC!
    openmc.run()


def process_aft_openmc(folder1='random_savearray/', file1='detector_1source_20220118.txt', \
                        folder2='random_savefig/', file2='detector_1source_20220118.png',\
                            source_x=100, source_y=100, norm=True):
    # We do not know how many batches were needed to satisfy the
    # tally trigger(s), so find the statepoint file(s)
    statepoints = glob.glob('statepoint.*.h5')

    # Load the last statepoint file
    sp = openmc.StatePoint(statepoints[-1])

    # Find the mesh tally with the StatePoint API
    tally = sp.get_tally(name='mesh tally')

    data = tally.get_values()#scores=['absorption'])

    # Get a pandas dataframe for the mesh tally data
    df = tally.get_pandas_dataframe(nuclides=False)

    # Set the Pandas float display settings
    pd.options.display.float_format = '{:.2e}'.format

    # Print the first twenty rows in the dataframe
    #df#.head(20)

    # Extract thermal absorption rates from pandas
    fiss = df[df['score'] == 'absorption']

    # Extract mean and reshape as 2D NumPy arrays
    mean = fiss['mean'].values.reshape((10,10)) # numpy array   #!20220118
    max = mean.max()
    if norm:
        mean_me = mean.mean()
        mean_st = mean.std()
        mean = (mean-mean_me)/mean_st

    absorb = tally.get_slice(scores=['absorption'])
    stdev = absorb.std_dev.reshape((10,10))
    stdev_max = stdev.max()

    #==================================
        
    data_json={}
    data_json['source']=[source_x, source_y]
    data_json['intensity']=100   #!20220119 tentative value!
    data_json['miu_detector']=0.3   #!20220119 constant!
    data_json['miu_medium']=1.2   #!20220119 constant!
    data_json['miu_air']=0.00018   #!20220119 constant!
    data_json['output']=get_output([source_x, source_y]).tolist()
    data_json['miu_de']=0.5   #!20220119 constant!
    mean_list=mean.T.reshape((1, 100)).tolist()
    data_json['input']=mean_list[0]  #!20220119 Notice!!!
    data_json['bean_num']=0.5   #!20220119 constant!
    modelinfo={'det_num_x': 10, 'det_num_y': 10, 'det_y': 0.03, 'det_x': 0.03, 'size_x': 0.5, 'size_x': 0.5, 'size_y': 0.5, 'med_margin': 0.0015}    #!20220119 constant!
    data_json['model_info']=modelinfo   #!20220119 constant!

    with open(folder1+file1,"w") as f:
        json.dump(data_json, f)
    
    #==================================
    print("mean_max:")
    print(max)
    print("stdev_max:")
    print(stdev_max)
    print("mean/stdev ratio:")
    print(max/stdev_max)

    plt.imshow(mean, interpolation='nearest', cmap="plasma")
    ds, ag = file2[:-5].split('_')
    plt.title('dist: ' + ds + ',  angle: ' + ag + '\nMean_max: ' + str(max) + '\nStdev_max: ' + str(stdev_max))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.savefig(folder2 + file2)
    plt.close()


def get_output(source):
    sec_center=np.linspace(-np.pi,np.pi,41)
    output=np.zeros(40)
    sec_dis=2*np.pi/40.
    angle=np.arctan2(source[1],source[0])
    before_indx=int((angle+np.pi)/sec_dis)
    after_indx=before_indx+1
    if after_indx>=40:
        after_indx-=40
    w1=abs(angle-sec_center[before_indx])
    w2=abs(angle-sec_center[after_indx])
    if w2>sec_dis:
        w2=abs(angle-(sec_center[after_indx]+2*np.pi))
    output[before_indx]+=w2/(w1+w2)
    output[after_indx]+=w1/(w1+w2)
    return output

if __name__ == '__main__':
    ###=================Input parameter======================
    num_data = 126
    batches = 100
    panel_density = 5.76 #g/cm3
    src_E = None
    src_Str = 10
    num_particles = 50000
    dist_min = 100
    dist_max = 1000
    angle = 0
    idx = 112
    energy_filter_range = [0.1e6, 2e6]
    e_filter_tf=False
    energy_a = 0.5e6
    energy_b = 1e6
    ###=======================================
    start = timeit.timeit()
    start_time = datetime.now()
    
    gen_materials_geometry_tallies(panel_density, e_filter_tf, energy_filter_range)
    j=batches
    for i in range(num_data):
        rad_dist=np.random.randint(dist_min, dist_max) + np.random.random(1)
        #rad_angle=np.random.randint(0, 359) + np.random.random(1)
        rad_angle=np.random.randint(0, 359)
        theta=rad_angle*np.pi/180
        #rad_source=[float(rad_dist*np.cos(theta)), float(rad_dist*np.sin(theta))]
        rad_x, rad_y=[float(rad_dist*np.cos(theta)), float(rad_dist*np.sin(theta))]
        print([rad_x, rad_y])
        get_output([rad_x, rad_y])

        gen_settings(src_energy=src_E, src_strength=src_Str,  en_a=energy_a, en_b=energy_b, num_particles=num_particles, batch_size=j, source_x=rad_x, source_y=rad_y)
        
        openmc.run(mpi_args=['mpiexec', '-n', '4', "-s", '11']) #!20220327

        folder1='save_data/'
        #file1=str(round(rad_dist[0], 5)) + '_' + str(round(rad_angle[0], 5)) + '.json'
        file1=str(round(rad_dist[0], 4)) + '_' + str(rad_angle) + '.json'
        folder2='save_figure/'
        #file2=str(round(rad_dist[0], 5)) + '_' + str(round(rad_angle[0], 5)) + '.png'
        file2=str(round(rad_dist[0], 4)) + '_' + str(rad_angle) + '.png'
        process_aft_openmc(folder1, file1, folder2, file2, rad_x, rad_y, norm=True)

    end = time.time()
    end_time = datetime.now()
    print("Start at " + str(start_time))
    print("Finish at " + str(end_time))
    time_s = end - start
    print("Total time [s]: " + str(time_s))
    print(time.strftime('%H:%M:%S', time.gmtime(time_s)))








