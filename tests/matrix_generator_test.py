from matrix_generator import initial_diag_matrix


def test_initial_diag_correct():
    diag_size, norm_value = 2, 10
    resulted_matrix = initial_diag_matrix(size=diag_size, norm_value=norm_value)

    assert resulted_matrix.shape == (diag_size,)


def test_initial_diag_correct_frac_part_correct():
    diag_size, norm_value = 10, 10
    resulted_matrix = initial_diag_matrix(size=diag_size, norm_value=norm_value)

    assert resulted_matrix.shape == (diag_size,)
