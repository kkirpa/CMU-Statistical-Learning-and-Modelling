import numpy as np
import math
from mpi4py import MPI
from matplotlib import pyplot as plt
import time

# Defining the size of the grid
ROWS, COLUMNS = 1000, 1000

# MAX_TEMP_ERROR is the convergence criterion — the maximum difference between iterations
# at which point we consider the temperature to be stable (i.e., the simulation can stop)
MAX_TEMP_ERROR = 0.01

# Initializing MPI communicator to enable inter-process communication
# note: comm is an object that helps coordinate parallel computations by allowing the processes to communicate
comm = MPI.COMM_WORLD

# pe_rank stores the rank (ID) of the current process, which will determine which part of the data each process handles
# note: Ranks start from 0!!    
pe_rank = comm.Get_rank()

# running_processes is the total number of processes running this simulation in parallel 
# This will dictate how the grid is divided across processes
running_processes = comm.Get_size()

# We can calculate how many rows of the temperature grid each process will be responsible for
# rows_per_process is the number of rows each process will handle, calculated by dividing
# the total number of rows (ROWS) by the number of running processes
rows_per_process = ROWS // running_processes

# remainder_rows will be any leftover rows after dividing the grid evenly
# If the total number of rows isn't perfectly divisible by the number of processes,
# some processes will handle one additional row (remainder_rows keeps track of these extra rows)
remainder_rows = ROWS % running_processes

# First, we can calculate how many rows each process will handle (rows_per_process) 
# If there are any leftover rows that can't be evenly divided, some processes will handle 1 extra row
extra_rows = 1 if pe_rank < remainder_rows else 0

# The starting row for each process can be determined by multiplying the process rank (pe_rank) by the number of rows 
# each process is supposed to handle. We add any extra rows (from the remainder) to processes with a lower rank
start_row = pe_rank * rows_per_process + min(pe_rank, remainder_rows) + 1

# The ending row is going to be simply the starting row plus the number of rows this process handles, minus one
# If this process is one of the ones that gets an extra row, we add that as well
end_row = start_row + rows_per_process - 1 + extra_rows

# Each process creates a local temperature array, where it will store the temperature values for 
# the rows it is responsible for. We add 2 extra rows to each process' array for "ghost rows"
# that hold boundary conditions from neighboring processes
temperature_last = np.empty((rows_per_process + 2, COLUMNS + 2)) 
temperature = np.empty((rows_per_process + 2, COLUMNS + 2))

# Debugging statement that prints the range of rows each process is handling
print(f"PE {pe_rank}: start_row = {start_row}, end_row = {end_row}")

# Function to initialize the temperature grid
# This function is responsible for setting the initial conditions for the simulation
def initialize_temperature(temp):

    # Initially, setting all the temperatures in the grid to zero
    temp[:,:] = 0

    # Next, we set the boundary conditions for the right side of the grid
    # We do this by assigning a temperature value to the far-right column (COLUMNS+1)
    # based on the sine function, which simulates a curved temperature profile
    for i in range(rows_per_process + 2):
        # We calculate the global row index (the actual row in the global grid), 
        # which depends on the rank of the process handling this portion of the grid
        global_ind = i + pe_rank * rows_per_process
        # Applying the boundary condition for the right edge
        temp[i, COLUMNS+1] = 100 * math.sin(((3.14159 / 2) / ROWS) * global_ind)

    # If this process is the last one (handling the bottom of the grid), we also set the
    # bottom boundary condition. The bottom boundary is assigned values similar to the
    # right side, using a sine wave pattern, but this time applied along the bottom row
    if pe_rank == running_processes - 1:
        for j in range(COLUMNS + 2):
            temp[rows_per_process + 1, j] = 100 * math.sin(((3.14159 / 2) / COLUMNS) * j)

# Marking the start time to measure the overall performance of the simulation
start_time = time.time()

# Function to output the final temperature grid as an image
def output(data):
    # Using matplotlib to create a visual representation of the temperature distribution
    # I saved it as a image so I can visually inspect the results
    plt.imshow(data)
    plt.savefig("plate.jpeg")
    print("Saved image as plate.jpeg")

# calling the function to initialize the temperature grid
# This sets the initial conditions for the temperature values in the grid
initialize_temperature(temperature_last)

# The next step is to broadcast the maximum number of iterations to all processes
# PE 0 (the process with rank 0) prompts the user for input on the maximum number of iterations
# the simulation should run. This value is then sent to all other processes
if pe_rank == 0:
    max_iterations = int(input("Maximum iterations: "))
else:
    max_iterations = None

# Broadcasting the value of max_iterations to all processes, ensuring everyone has the same value
max_iterations = comm.bcast(max_iterations, root=0)

# Setting the initial temperature difference and iteration counter
# dt the maximum temperature difference between iterations? used to check for convergence
dt = 100
iteration = 1

# Starting the main simulation loop, which runs until the temperature grid converges
# (i.e., the temperature changes by less than MAX_TEMP_ERROR between iterations) or we hit the maximum number of iterations
while dt > MAX_TEMP_ERROR and iteration < max_iterations:

    # For each iteration, we will update the temperature at the internal grid points
    # The update formula averages the temperature from neighboring cells, simulating heat diffusion
    for i in range(1, rows_per_process + 1):
        for j in range(1, COLUMNS + 1):
            temperature[i, j] = 0.25 * (temperature_last[i + 1, j] + temperature_last[i - 1, j] +
                                        temperature_last[i, j + 1] + temperature_last[i, j - 1])
    
    # Initializing the local temperature difference for each process, which will later be used to 
    # determine if the system has reached a stable state (i.e., converged)
    local_dt = 0

    # To ensure that boundary conditions are consistent between neighboring processes, we will perform  ghost 
    # row exchanges. Processes send and receive their boundary rows to/from their neighboring processes 

    # If the process has a rank greater than 0, it sends its top boundary row to the process above 
    # and receives a row from that process
    if pe_rank > 0:  
        comm.Send(temperature_last[1, :], dest=pe_rank - 1, tag=0)
        comm.Recv(temperature_last[0, :], source=pe_rank - 1, tag=1)

    # If the process has a rank less than the total number of processes - 1, it sends its bottom
    # boundary row to the process below and receives a row from that process
    if pe_rank < running_processes - 1:  
        comm.Send(temperature_last[rows_per_process, :], dest=pe_rank + 1, tag=1)
        comm.Recv(temperature_last[rows_per_process + 1, :], source=pe_rank + 1, tag=0)

    # After exchanging boundary information, we calculate the maximum local temperature change
    # for this process by comparing the new temperature values with the previous iteration
    local_dt = np.max(np.abs(temperature[1:rows_per_process+1, 1:COLUMNS+1] - temperature_last[1:rows_per_process+1, 1:COLUMNS+1]))

    # We update the local temperature grid for the next iteration by copying the newly
    # calculated values into the temperature_last array
    temperature_last[1:rows_per_process+1, 1:COLUMNS+1] = temperature[1:rows_per_process+1, 1:COLUMNS+1]

    # To determine the global temperature difference, we perform an all-reduce operation
    # This combines the maximum temperature difference from all processes and stores the result in dt
    dt = comm.allreduce(local_dt, op=MPI.MAX)

    # Every iteration, process 0 (PE 0) prints the current iteration number and the maximum error (difference in temp?)
    if pe_rank == 0:
        print(f"Iteration {iteration}, max error = {dt}", flush=True)

    iteration += 1

# After the simulation is complete, we gather the final temperature arrays from all processes
# Only PE 0 (rank 0) collects the full temperature grid, while the other processes send their local data
if pe_rank == 0:
    full_temperature_last = np.empty((ROWS, COLUMNS)) 
    # PE 0 allocates space for the full grid 
else:
    full_temperature_last = None

# Gathering the temperature data from all processes into full_temperature_last on PE 0
comm.Gather(temperature_last[1:-1, 1:-1].copy(), full_temperature_last, root=0)

# Once the grid is gathered, PE 0 outputs the final temperature grid as an image and I am printing its dimensions for debugging
if pe_rank == 0:
    print(f"Full temperature matrix dimensions: {full_temperature_last.shape}")
    output(full_temperature_last)

# Marking the end time and printing the total execution time
end_time = time.time()  
if pe_rank == 0:
    print(f"Execution time: {end_time - start_time} seconds")

MPI.Finalize()
