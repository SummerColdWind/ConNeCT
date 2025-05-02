def get_schedule_jump(T, N, M,
                      jump2_length=1, jump2_n_sample=1,
                      jump3_length=1, jump3_n_sample=1,
                      start_resampling=100000000):

    def init_jumps(length, count):
        return {j: count - 1 for j in range(0, T - length, length)}

    def apply_jump(t, jump_dict, length, reset_dicts=None):
        if jump_dict.get(t, 0) > 0 and t <= start_resampling - length:
            jump_dict[t] -= 1
            for _ in range(length):
                t += 1
                ts.append(t)
            if reset_dicts:
                for reset_func in reset_dicts:
                    reset_func()
        return t

    # Initialize jump schedules
    jumps = init_jumps(N, M)
    jumps2 = init_jumps(jump2_length, jump2_n_sample)
    jumps3 = init_jumps(jump3_length, jump3_n_sample)

    def reset_jumps2():
        nonlocal jumps2
        jumps2 = init_jumps(jump2_length, jump2_n_sample)

    def reset_jumps3():
        nonlocal jumps3
        jumps3 = init_jumps(jump3_length, jump3_n_sample)

    t = T
    ts = []

    while t >= 1:
        t -= 1
        ts.append(t)
        t = apply_jump(t, jumps3, jump3_length)
        t = apply_jump(t, jumps2, jump2_length, reset_dicts=[reset_jumps3])
        t = apply_jump(t, jumps, N, reset_dicts=[reset_jumps2, reset_jumps3])

    ts.append(-1)
    return ts


if __name__ == '__main__':
    print(len(get_schedule_jump(250, 10, 10)))

