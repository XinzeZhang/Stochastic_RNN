    # load data and make training set
    data = torch.load('traindata.pt')

    train_input = data[:-1, :-1]
    train_target = data[:-1, 1:]

    # Simple training
    model_esn = SimpleESN(n_readout=1000, n_components=1000,
                          damping=0.3, weight_scaling=1.25)
    echo_train_state = model_esn.fit_transform(train_input)
    regr = Ridge(alpha=0.01)
    regr.fit(echo_train_state, train_target)

    echo_train_target, echo_train_pred= train_target, regr.predict(echo_train_state)

    err = mean_squared_error(echo_train_target, echo_train_pred)

    data_figures = plt.figure(figsize=(12, 4))
    trainplot = data_figures.add_subplot(1, 3, 1)
    trainplot.plot(train_input[1,:], 'b')
    trainplot.set_title('training signal')

    echoplot = data_figures.add_subplot(1, 3, 2)
    echoplot.plot(echo_train_state[:, :20])
    echoplot.set_title('Some reservoir activation')

    testplot = data_figures.add_subplot(1, 3, 3)
    testplot.plot(train_target[1,:], 'b', label='test signal')
    testplot.plot(echo_train_pred[1,:], 'g', label='prediction')
    testplot.set_title('Prediction (MSE %0.3f)' % err)

    testplot.legend(loc='lower right')
    plt.tight_layout(0.5)
    plt.savefig('Data_Prediction.png')