from random import choice
from numpy import array, dot, random

unit_step = lambda x: 0 if x < 0 else 1

training_data = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 1),
]

training_matrices = [
    # Template:
    # (array([[0,0,],[0,1,],[0,2,],[1,0,],[1,1,],[1,2,],[2,0,],[2,1,]]), 1)
    (array([[0,0,1],[0,1,0],[0,2,1],[1,0,0],[1,1,1],[1,2,0],[2,0,0],[2,1,1]]), [2,2,1])
]

training_set = [
    (array([1,2,0,1,0,0,0,0,2]), 6),
    (array([1,2,0,0,1,0,2,0,0]), 7),
    (array([1,0,2,0,1,0,0,0,2]), 5),
    (array([0,0,0,0,0,0,0,0,0]), 4), # Empty board
    (array([0,0,0,0,2,0,0,0,0]), 5), # One cross at center
    (array([1,0,0,2,2,1,1,0,0]), 1),
    (array([0,0,1,0,2,0,1,0,2]), 0),
    (array([0,0,0,0,0,2,0,1,1]), 6),
    (array([1,1,2,1,2,0,0,0,0]), 6),
    (array([1,0,2,0,2,1,0,0,0]), 6),
    (array([1,1,2,2,1,1,0,2,0]), 8)
]

"""
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), ),
    (array([,,,,,,,,]), )
"""

w = random.rand(9)
errors = []
eta = 0.2
n = 100

for position in training_set:
    #print(dot(position[1], w))
    pass


for i in xrange(n):
    x, expected = choice(training_set) # Separating input values and expected ans
    # print("X is:", x, "; expected is:", expected)
    result = dot(w, x)
    error = expected - unit_step(result)
    #errors.append(error)
    w += eta * error * x


print("Trained weights:", w)
for x, expected in training_set:
    result = dot(x, w)
    print("{}: expected{}; {} -> {}".format(x, expected, result, unit_step(result)))
