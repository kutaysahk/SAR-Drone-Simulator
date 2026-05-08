from state.environment import EnvironmentState
from state.drone import DroneState
from ui.visualizer import Visualizer, show_start_screen
from optimization.genetic_algo import GeneticOptimizer

def run_simulation():
    GRID_WIDTH = 60
    GRID_HEIGHT = 40
    CELL_SIZE = 15

    while True:
        run_scenario(GRID_WIDTH, GRID_HEIGHT, CELL_SIZE)

def run_scenario(grid_width: int, grid_height: int, cell_size: int):
    show_start_screen()
    
    env = EnvironmentState(width=grid_width, height=grid_height)
    env.generate_disaster_zone(obstacle_density=150, hazard_density=40, num_survivors=8)
    
    drone = DroneState(start_x=0, start_y=0, max_battery=1500,
                       env_width=grid_width, env_height=grid_height)
                       
    # --- PILLAR 3: RUN OPTIMIZATION ---
    optimizer = GeneticOptimizer(population_size=50)
    optimized_weights = optimizer.run_evolution(generations=100)
    
    ui = Visualizer(env_width=grid_width, env_height=grid_height, cell_size=cell_size)
    print("Starting Project Aegis: Simulation...")

    changed_cells = drone.sense_environment(env)
    ui.animate_reveal(env, drone, changed_cells)

    running = True
    while running and drone.is_active and not ui.return_to_menu:
        # Replan every step so newly sensed nearby gaps are handled before
        # continuing toward an older, farther target.
        nearest_path = drone.find_path_to_nearest_frontier(env, optimized_weights)

        if nearest_path:
            next_x, next_y = nearest_path[0]
            ui.animate_move(env, drone, next_x, next_y)
            if ui.return_to_menu:
                break
            if drone.move_to(next_x, next_y):
                changed_cells = drone.sense_environment(env)
                ui.animate_reveal(env, drone, changed_cells)
        else:
            print("Search complete: no reachable unchecked frontier remains.")
            running = False

    if ui.return_to_menu:
        print("Returning to start screen.")
        return

    print("Simulation Ended.")
    print("Recent logic proof trace:")
    for proof_step in drone.kb.explain_recent_inferences():
        print(f"- {proof_step}")

if __name__ == "__main__":
    run_simulation()
