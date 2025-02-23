import jax
import jax.numpy as jnp


def ceil_div(n, d):
	return -(n // -d)


def zero_prefix(A, index):
	mask = jnp.arange(A.shape[0]) >= index
	return A * mask


def fill_suffix(A, index, val):
	mask = jnp.arange(A.shape[0]) >= index
	return jnp.where(mask, val, A)


def replace(A, val1, val2):
	return jnp.where(A == val1, val2, A)


def place_2D(R, M, P):
	"""Assign values from P into R at the row indices indicated by the 1D mask M."""
	M = jnp.broadcast_to(jnp.expand_dims(M, -1), M.shape + (2,))
	return jnp.place(R, M, P, inplace=False)


def ith_nonzero_index(A, i):
	"""Gets the index of the ith nonzero element of a 1D boolean array 'A'."""
	cumsums = jnp.cumsum(A)
	return jnp.searchsorted(cumsums, i + 1, side='left')	# change method


def unravel_2Dindex(i, dim2):
	"""Convert a single raveled index 'i' into 2D indices under shape (dim1, dim2)."""
	return jnp.array((i // dim2, i % dim2))


def remove_rows_jit(A, I, pad_value=-1):
	k = A.shape[0]
	I = jnp.where(I == pad_value, 2*k, I)
	keep_mask = jnp.ones((k,), dtype=bool)
	keep_mask = keep_mask.at[I].set(False, mode="drop")
	keep_indices = jnp.where(keep_mask, size=k, fill_value=k+1)
	A_minus = A.at[keep_indices].get(mode="fill", fill_value=pad_value)
	return A_minus


def expand_partition(P, part_size, pad_value):
	"""Convert a 1D coloring into a 2D partition."""
	k = P.shape[0]

	order = jnp.argsort(P, stable=True)
	P_sorted = P[order]

	part_positions = jnp.arange(k) - jnp.searchsorted(P_sorted, P_sorted)

	partition = jnp.full((k, part_size), pad_value)
	partition = partition.at[P_sorted, part_positions].set(order)
	return partition


def rearrange_padding(A, pad_value):
	"""
	Given a 2D pad_value-padded array A, rearranges each row to put all padding
	after all other elements while preserving the order of those elements.
	"""
	d = A.shape[1]

	def rearrange_row(x):
		indices = jnp.arange(d)
		valid_key = x == pad_value
		compound_key = (valid_key * d) + indices
		order = jnp.argsort(compound_key)
		return x[order]

	return jax.vmap(rearrange_row)(A)


def compute_group_sums(A, B, num_groups):
	"""Sums the values of A for each unique value in B and returns the sums indexed by B."""
	group_sums = jax.ops.segment_sum(A, B, num_groups)
	return group_sums[B]


def normalize(A):
	"""Normalize an array A of vectors."""
	divisors = jnp.linalg.vector_norm(A, axis=-1, keepdims=True)
	return jnp.where(divisors == 0.0, A, A / divisors) 


def discrete_ball_map(indices):
	map_matrix = jnp.array([[1, -1], [1, 1]])
	return (indices @ map_matrix.T ) // 2	# divide by 2 after for possible efficiency


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
	k = edges.shape[0]

	visited = jnp.zeros(k, dtype=bool)
	visited = visited.at[i].set(True)
	queue = jnp.full((k,), pad_value, dtype=int)
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


def smallest_missing(x, num_colors, pad_value):
	# assumes there is always an unused color
	mask = jnp.zeros(num_colors, dtype=bool)
	mask = mask.at[x].set(True, mode="drop")
	return jnp.argmin(mask)


def greedy_graph_coloring(edges, pad_value):
	k = edges.shape[0]
	colors = jnp.full(k, pad_value)

	def color_vertex(i, colors):
		neighbors = edges[i]
		valid = (neighbors != pad_value) & (neighbors < i)
		safe_nbrs = jnp.where(valid, neighbors, 2*k)
		nbr_colors = colors.at[safe_nbrs].get(mode="fill", fill_value=2*k)
		choice = smallest_missing(nbr_colors, k, pad_value)
		colors = colors.at[i].set(choice)
		return colors

	colors = jax.lax.fori_loop(0, k, color_vertex, colors)
	return colors