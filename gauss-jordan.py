import numpy as np

def find_pivot_row(column, i):
    return np.argmax(column[i:] != 0) + i

def clean_column(matrix, pivot_position):
    row_count, _ = matrix.shape
    # print(f'pivot position: {pivot_position}')
    matrix[pivot_position[0]] /= matrix[pivot_position]

    for i in range(row_count):
        if i == pivot_position[0] or matrix[i, pivot_position[1]] == 0:
            continue
        matrix[i] -= matrix[i, pivot_position[1]] * matrix[pivot_position[0]]

def gauss_jordan(coef_matrix, aug_matrix=None):
    augmented_matrix = np.concatenate((coef_matrix, aug_matrix), axis=1)
    row_cursor = 0
    row_count, col_count = augmented_matrix.shape
    for j in range(col_count):
        if row_cursor >= row_count:
            break
        i = find_pivot_row(augmented_matrix[:, j], row_cursor)
        if i != row_cursor:
            augmented_matrix[[row_cursor, i]] = augmented_matrix[[i, row_cursor]]
        if augmented_matrix[(row_cursor, j)] == 0:
            continue

        clean_column(augmented_matrix, (row_cursor, j))
        # print(augmented_matrix)
        row_cursor += 1

    return augmented_matrix


n = int(input())
A = np.zeros(shape=(n, n+1))
for i in range(n):
    new_row = list(map(int, input().split(' ')))
    A[i] = new_row


rref = gauss_jordan(A[:, :-1], A[:, -1:])
# print(rref)

if np.trace(rref) == n:
    print(np.sort(rref[:, -1]))
else:
    print('no unique solution')
