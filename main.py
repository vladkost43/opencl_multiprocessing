import time

import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy as np  # Import Numpy number tools

context = cl.create_some_context()  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

a = pycl_array.to_device(queue, np.random.rand(500000).astype(np.float32))
b = pycl_array.to_device(queue, np.random.rand(500000).astype(np.float32))
# Create two random pyopencl arrays
c = pycl_array.empty_like(a)  # Create an empty pyopencl destination array
process_count = int(input("Write precesses count = "))
timer = time.time()
program = cl.Program(context, f"""
__kernel void sum(__global const float *a, __global const float *b, __global float *c)
{{
  int i = get_global_id({process_count});
  c[i] = (a[i]*2 + b[i]/3)/90 - (a[i]*0.33+ b[i]*4.75)*78;
}}""").build()  # Create the OpenCL program

program.sum(queue, a.shape, None, a.data, b.data, c.data)  # Enqueue the program for execution and store the result in c

print("a: {}".format(a))
print("b: {}".format(b))
print("c: {}".format(c))
print(time.time() - timer)
# Print all three arrays, to show sum() worked
