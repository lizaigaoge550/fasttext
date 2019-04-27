def print_size(**kwargs):
    for key, value in kwargs.items():
        print(f'{key} : {value.size()}')