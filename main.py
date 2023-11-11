# import mayavi.mlab
# import numpy as np
# import random

# shape = (100, 100, 100)
# data = np.zeros(shape)

# for x in range(shape[0]):
#     for y in range(shape[1]):
#         for z in range(shape[2]):
#             output = np.random.rand()
#             if output > 0.7:
#                 data[x, y, z] = 1

# # data[0:50, 50:70, 0:50] = 1
# # data[0:50, 0:20, 0:50] = 1

# xx, yy, zz = np.where(data == 1)


# mayavi.mlab.points3d(xx, yy, zz,
#                      mode="cube",
#                      color=(0, 1, 0),
#                      scale_factor=1)

# mayavi.mlab.show()


import visualize
import numpy as np
import pyvista as pv
import neat
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vtk 

from skimage import measure
n = 20


def eval_genomes(genomes, config):
    global p
    all_genome_ids = []
    all_genomes = []
    idx = 0
    for (genome_id, genome) in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        voxel_grid = np.zeros((n, n, n))

        for x in range(n):
            for y in range(n):
                for z in range(n):
                    d = np.sqrt((x)**2 + (y)**2 + (z)**2)
                    val = net.activate([x, y, z, d])[0]
                    voxel_grid[x, y, z] = val

        
                
        
        
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        # # Create an empty mesh
        # mesh = pv.PolyData()

        # # Iterate through the boolean array and create cubes for True values
        # for i in range(voxel_grid.shape[0]):
        #     for j in range(voxel_grid.shape[1]):
        #         for k in range(voxel_grid.shape[2]):
        #             if voxel_grid[i, j, k]:
        #                 # Create a cube at the corresponding location
        #                 cube = pv.Cube(center=(i, j, k))
        #                 # Combine the cube with the mesh
        #                 if mesh.n_cells == 0:
        #                     mesh = cube
        #                 else:
        #                     mesh += cube

        # if mesh.number_of_points == 0 or voxel_grid.sum() >= 0.9 * (n**3):
        #     print(f'{voxel_grid.sum()} > 0.9 * {n ** 3}')
        #     genome.fitness -= 1000000
        # else:

        #     # Now apply smoothing using vtkSmoothPolyDataFilter
        #     smoother = vtk.vtkSmoothPolyDataFilter()
        #     smoother.SetInputData(mesh)
        #     smoother.SetNumberOfIterations(100)  # Set the number of smoothing iterations
        #     smoother.Update()
        #     mesh = pv.wrap(smoother.GetOutput())

        #     # Visualization
        #     p = pv.Plotter()
        #     p.add_mesh(mesh)
        #     p.show()

        #     p = pv.Plotter()
        #     p.add_mesh(mesh)  # Assume smoothed_mesh is your final mesh
        #     p.open_gif(f"rotation{genome_id}.gif")

        #     focal_point = (n/2, n/2, n/2)
        #     for i in range(0, 360, n // 2):  # Adjust the range and step to control the rotation speed
        #         p.camera_position = [(3*n*np.sin(np.radians(i)) + n/2, 3*n*np.cos(np.radians(i)) + n/2, n + n / 2), focal_point, (0, 0, 1)]
        #         p.write_frame()  # Write this frame to the GIF

        #     p.close()  # Close the plotter and finalize the GIF





        # Parameters
        x_min, y_min, z_min = -n, -n, -n
        x_max, y_max, z_max = n, n, n

        # Create a uniform grid to sample the function with
        grid = pv.ImageData(
            dimensions=(n, n, n),
            spacing=((x_max - x_min) / (n - 1), (y_max - y_min) / (n - 1), (z_max - z_min) / (n - 1)),
            origin=(x_min, y_min, z_min),
        )

        voxel_grid = np.array(voxel_grid)

        # Set the random voxel data as the scalar data for the grid
        grid.point_data["Values"] = voxel_grid.flatten(order='F')

        # Extract a contour from the grid using the Marching Cubes algorithm
        mesh = grid.contour([0.4], scalars="Values", method='marching_cubes')  # Use 0.5 as the threshold value

        # Compute distances for coloring
        dist = mesh.points

        if mesh.number_of_points == 0:
            genome.fitness -= 1000000
        else:
            plotter = pv.Plotter(off_screen=True) 
            # Specify the file path for the GIF
            # plotter.open_gif(f"rotation{genome_id}.gif")
            
            # Add the mesh to the plotter
            plotter.add_mesh(mesh, scalars=dist, smooth_shading=True, specular=1, cmap="plasma")
            
            # # Create a loop to rotate the camera and capture frames
            # n_frames = 36  # Number of frames
            # for i in range(n_frames):
            #     sign = 1 if i < n_frames // 2 else -1
            #     # Rotate the camera by 10 degrees at a time around the z-axis at (0,0,0)
            #     plotter.camera_position = [(8*n*np.sin(np.radians(i*10)), 8*n*np.cos(np.radians(i*10)), 2*n), (0,0,0), (0, 0, 1)]
            #     plotter.write_frame()  # Capture the frame
            
            # Close plotter to finalize the GIF
            plotter.close()
            
            # Plot the mesh
            mesh.plot(scalars=dist, smooth_shading=True, specular=1, cmap="plasma", show_scalar_bar=False)


p = None


def run(config_file):
    global p
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 300)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
