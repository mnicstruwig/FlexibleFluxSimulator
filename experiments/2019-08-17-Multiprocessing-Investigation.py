import ray

# ray.init()  # Initialize Ray, only call this once


def regular_function():
    return 1


@ray.remote
def remote_function():
    return 1


# Invocation is a little different now
result = regular_function()  # Works like regular function --> returns the result
obj_id = (
    remote_function.remote()
)  # Invocation for a remote function --> returns on object ID


remote_result = ray.get(obj_id)  # Fetching a remote result

# These happen serially
for _ in range(4):
    regular_function()

# But these happen in *parallel*
for _ in range(20):
    remote_function.remote()

# Note that we can also pass in Object IDs into remote functions (instead of
# fetching the results)

ready_ids, remaining_ids = ray.wait([obj_id])
