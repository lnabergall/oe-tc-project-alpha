import jax
import jax.numpy as jnp


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