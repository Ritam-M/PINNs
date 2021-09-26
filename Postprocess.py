def Postprocess(training_points):
    test_X = [[], [], []]
    test_Y = [[], [], []]

    for idx, sample in enumerate(training_points):

        t, x = sample
        if t == 0.25:
            test_X[0].append(sample)
            test_Y[0].append(u_data_transformed[idx])

        if t == 0.50:
            test_X[1].append(sample)
            test_Y[1].append(u_data_transformed[idx])        

        if t == 0.75:
            test_X[2].append(sample)
            test_Y[2].append(u_data_transformed[idx])
            
    return (test_X, test_Y)
