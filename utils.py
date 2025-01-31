import jax
import jax.numpy as jnp


def place_2D(R, M, P):
	"""Assign values from P into R at the row indices indicated by the 1D mask M."""
	M = jnp.broadcast_to(jnp.expand_dims(M, -1), M.shape + (2,))
	return jnp.place(R, M, P, inplace=False)


def ith_nonzero_index(A, i):
	"""Gets the index of the ith nonzero element of a 1D boolean array 'A'."""
	cumsums = jnp.cumsum(A)
	return jnp.searchsorted(cumsums, i + 1, side='left')


def unravel_2Dindex(i, dim2):
	"""Convert a single raveled index 'i' into 2D indices under shape (dim1, dim2)."""
	return jnp.array((i // dim2, i % dim2))


def remove_rows_jit(A, I, pad_value=-1)
	k = A.shape[0]
	I = jnp.where(I == pad_value, 2*k, I)
	keep_mask = jnp.ones((k,), dtype=bool)
	keep_mask = keep_mask.at[I].set(False, mode="drop")
	keep_indices = jnp.where(keep_mask, size=k, fill_value=k+1)
	A_minus = A.at[keep_indices].get(mode="fill", fill_value=pad_value)
	return A_minus


def normalize(A):
	"""Normalize an array A of vectors."""
	return A / jnp.linalg.vector_norm(A, axis=-1, keepdims=True)


def discrete_ball_map(indices):
	map_matrix = jnp.array([[1, -1], [1, 1]])
	return (indices @ map_matrix.T ) / 2	# divide by 2 after for possible efficiency


def bfs(edges, i, pad_value):
	"""
	Perform a breadth-first search on the array 'edges'. 
	The ith row of 'edges' is the neighborhood of vertex i. 

	edges
		Array of vertices with padding. 2D.
	i
		Starting vertex.
	pad_value
		Scalar used as a padding used to indicate elements of edges that should be treated as null.
	"""
	num_vertices = row_indices.shape[0]

	visited = jnp.zeros(num_vertices, dtype=bool)
	visited = visited.at[i].set(True)
	queue = jnp.full((num_vertices,), pad_value, dtype=int)
	queue = queue.at[0].set(i)
	head, tail = 0, 1

	def cond_fn(state):
		_, _, head, tail = state
		return head < tail

	def body_fn(state):
		visited, queue, head, tail = state
		current = queue[head]
		head += 1

		neighbors = edges[current]

		def process_neighbor(state, neighbor):
			visited, queue, tail = state
			valid = (neighbor != pad_value) & (~visited[neighbor])
			visited = jnp.where(valid, visited.at[neighbor].set(True), visited)
			queue = jnp.where(valid, queue.at[tail].set(neighbor), queue)
			tail = jnp.where(valid, tail + 1, tail)
			return (visited, queue, tail), pad_value

		(visited, queue, tail), _ = jax.lax.scan(process_neighbor, (visited, queue, tail), neighbors)
		return visited, queue, head, tail

	visited, queue, head, tail = jax.lax.while_loop(cond_fn, body_fn, (visited, queue, head, tail))
	return visited