import numpy as np

np.random.seed(seed=24)

random_integers = np.random.randint(low=1,high=5,size=100)

print(random_integers[:5])

#locais em que o random_integers é igual 3

is_equal_to_3 = random_integers == 3

print(is_equal_to_3[:5])

#soma da máscara booleana
print(sum(is_equal_to_3))

#máscara booleana pode indexar arrays
print(random_integers[is_equal_to_3])
