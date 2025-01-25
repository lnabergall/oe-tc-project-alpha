import jax
import jax.numpy as jnp


@jax.jit
def lattice_distance(r, s):
	"""Compute lattice distance between points r and s."""
	return jnp.linalg.norm(r - s, ord=1, axis=-1)


@jax.jit
def lattice_distances(r, S):
	"""
	Compute lattice distance between point r and every point in the array S.
	The last dimension of S should have the same size as r. 
	"""
	return jnp.linalg.norm(r - S, axis=-1)


def canonical_shortest_path(r_start, r_end, n):
	"""
	Returns a canonical shortest path between r_start and r_end, where we take 
	a staircase path until we reach a straight line to r_end. 
	"""
	r_diff = r_end - r_start
	r_diff_sign = jnp.sign(r_diff)
	r_diff_abs = jnp.abs(r_diff)
	length = jnp.sum(r_diff_abs)
	i_square_half = jnp.abs(jnp.min(r_diff))
	i_square *= 2
	element_square = r_start + (r_diff_sign * jnp.array((i_square_half, i_square_half)))
	step_outside_square = jnp.where(
		r_diff_abs[0] >= r_diff_abs[1], jnp.array([1, 0]), jnp.array([0, 1]))
	padding = jnp.array((-1, -1), dtype=int)

	def path_creator(i, j):
		half, parity = i // 2, i % 2
		i_diff = i - i_square
		element_within_square = r_start + (r_diff_sign * jnp.array((half + parity, half)))
		element_outside_square = element_square + (i_diff * step_outside_square)
		element_on_path = jax.lax.select_n(
			i <= i_square, element_outside_square, element_within_square)
		element = jax.lax.select_n(i > length, element_on_path, padding)
		return element

	path = jnp.fromfunction(path_creator, (n, 2), dtype=int)
	return path


def shift_path(path, r): 
	return path + r - path[0]


def append_on_path(path, r):
	insert_id = jnp.argmin(path, axis=0)[0]
	return jnp.insert(path, insert_id, r, axis=0)


def lattice_positions(n):
	return jnp.stack(jnp.indices((n, n)), axis=-1)


def boundaries(r, delta):
	#### THIS IS WRONG, NOT PERIODIC
	x, y = r
	min_x = jnp.clip(x - delta, min=0)
	min_y = jnp.clip(y - delta, min=0)
	max_x, max_y = x + delta, y + delta
	return min_x, max_x, min_y, max_y


def centered_square(L, r, delta):
	"""
	Generate the subset of the lattice 'L' consisting of a square of radius 'delta' centered at 'r',
	along with the positions in that subset.  
	"""
	min_x, max_x, min_y, max_y = boundaries(r, delta)
	square = L[min_x: max_x+1, min_y: max_y+1]
	x_positions = jnp.arange(min_x, max_x+1)
	y_positions = jnp.arange(min_y, max_y+1)
	positions = jnp.stack(jnp.meshgrid(x_indices, y_indices, indexing="ij"), axis=-1)
	return square, positions


@partial(jax.jit, static_argnums=[3])
def generate_lattice(R, n, pad_value, sites=2):
	"""
	Generate the occupation lattice corresponding to R. The lattice is nxnxsites, 
	with 'sites' slots per position of the lattice, to accomodate test positions; 
	occupation is indicated by a particle index in one of the slots. 
	"""
	k = R.shape[0]
	L = jnp.full((n, n, sites), pad_value, dtype=int)
	counts = jnp.zeros((n, n), dtype=int)

	def particle_fn(L, counts, i, x, y):
		idx = counts[x, y]
		L = L.at[x, y, idx].set(i)
		counts = counts.at[x, y].add(1)
		return (L, counts)

	def padding_fn(L, counts, *args):
		return L, counts

	def map_point(i, args):
		L, counts = args
		x, y = R[i]
		return jax.lax.cond(x == pad_value, padding_fn, particle_fn, L, counts, i, x, y)

	L, counts = jax.lax.fori_loop(0, k, map_point, (L, counts))
	return L, counts


def add_to_lattice(L, I, R, pad_value):
	"""
	Add particles in I to lattice L. Assumes 2-particle co-location limit is obeyed
	and no particles in I are co-located.
	"""
	def add(L, i):
		x, y = R[i]
		idx = jnp.argmax(L[x, y] == pad_value)
		L = L.at[x, y, idx].set(i)
		return L

	def update_fn(L, i):
		padding_fn = lambda L, _: L
		return jax.lax.cond(i == pad_value, padding_fn, add, L, i)

	L = jax.vmap(update_fn, in_axes=(None, 0), out_axes=None)(L, I)
	return L


def remove_from_lattice(L, I, R, pad_value):
	"""Remove particles in I from lattice L. Assumes that R matches L."""
	def remove(L, i):
		x, y = R[i]
		idx = jnp.argmax(L[x, y] == i)
		L = L.at[x, y, idx].set(pad_value)
		return L

	def update_fn(L, i):
		padding_fn = lambda L, _: L
		return jax.lax.cond(i == pad_value, padding_fn, remove, L, i)

	L = jax.vmap(update_fn, in_axes=(None, 0), out_axes=None)(L, I)
	return L


def unlabel_lattice(L, pad_value):
	return jnp.sum(L != pad_value, axis=-1)


def generate_unlabeled_lattice(R, n):
	"""
	Generate the unlabeled occupation lattice corresponding to R. The lattice is an nxn Array; 
	occupation is indicated by 0, 1, or 2, indicating the number of particles in the site. 
	"""
	L = jnp.zeros((n, n), dtype=int)
	L = L.at[R[:, 0], R[:, 1]].add(1)
	return L


def generate_property_lattice(L, C, pad_value):
	"""Gather property values from a 1D Array C using indices in an Array L while ignoring padding."""
	return jnp.where(L != pad_value, C[L], pad_value)