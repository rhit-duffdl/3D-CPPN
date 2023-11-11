import numpy as np
import pyvista as pv
import neat
import os

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
                    d = np.sqrt((x) ** 2 + (y) ** 2 + (z) ** 2)
                    val = net.activate([x, y, z, d])[0]
                    voxel_grid[x, y, z] = val

        # Parameters
        x_min, y_min, z_min = -n, -n, -n
        x_max, y_max, z_max = n, n, n

        # Create a uniform grid to sample the function with
        grid = pv.ImageData(
            dimensions=(n, n, n),
            spacing=(
                (x_max - x_min) / (n - 1),
                (y_max - y_min) / (n - 1),
                (z_max - z_min) / (n - 1),
            ),
            origin=(x_min, y_min, z_min),
        )

        voxel_grid = np.array(voxel_grid)

        grid.point_data["Values"] = voxel_grid.flatten(order="F")

        mesh = grid.contour(
            [0.0], scalars="Values", method="marching_cubes"
        )  # Use 0.5 as the threshold value

        dist = mesh.points

        if mesh.number_of_points == 0:
            genome.fitness -= 1000000
        else:
            plotter = pv.Plotter(off_screen=True)

            plotter.add_mesh(
                mesh, scalars=dist, smooth_shading=True, specular=1, cmap="plasma"
            )

            plotter.open_gif(f"rotation{genome_id}.gif")
            n_frames = 36
            for i in range(n_frames):
                sign = 1 if i < n_frames // 2 else -1
                # Rotate the camera by 10 degrees at a time around the z-axis at (0,0,0)
                plotter.camera_position = [
                    (
                        8 * n * np.sin(np.radians(i * 10)),
                        8 * n * np.cos(np.radians(i * 10)),
                        2 * n,
                    ),
                    (0, 0, 0),
                    (0, 0, 1),
                ]
                plotter.write_frame()

            plotter.close()

            mesh.plot(
                scalars=dist,
                smooth_shading=True,
                specular=1,
                cmap="plasma",
                show_scalar_bar=False,
            )
        idx += 1


p = None


def run(config_file):
    global p
    # Load configuration.
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 300)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config")
    run(config_path)
