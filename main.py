from utils.all import *
from setting import define_parser
from southwest_module import *

"""
Main processing
auth: Methodfunc - Kwak Piljong
date: 2021.08.27
modify date: 2021.08.30
version: 0.2
describe: makerset변경에 따른 코드 변경.
          train_loss, val_loss 수정
"""


def preprocessing_data(f_data, t_data, args):
    x_train, x_val, x_test = split_data(f_data, train_size=0.8, val_size=0.1, test_size=0.1)
    y_train, y_val, y_test = split_data(t_data, train_size=0.8, val_size=0.1, test_size=0.1)
    args.features_size = f_data.shape[-1]

    train_set = DataMaker(x=x_train, y=y_train, window_size=args.window_size, sliding=args.sliding_func)
    val_set = DataMaker(x=x_val, y=y_val, window_size=args.window_size, sliding=args.sliding_func)
    test_set = DataMaker(x=x_test, y=y_test, window_size=args.window_size, sliding=args.sliding_func)

    train = DataLoader(train_set, batch_size=args.batch_size, drop_last=True, shuffle=False)
    val = DataLoader(val_set, batch_size=args.batch_size, drop_last=True, shuffle=False)
    test = DataLoader(test_set, batch_size=args.batch_size, drop_last=True, shuffle=False)

    return train, val, test


if __name__ == '__main__':
    args = define_parser()

    # Load Data & Clean data
    raw_data = LoadDataframe(args.path).get_df()
    features, targets = cleanup_df(raw_data)

    # Scaling
    fscale, f_data = scaling(features)
    tscale, t_data = scaling(targets)
    args.features_size = f_data.shape[-1]

    # Preprocessing dataset
    trainloader, valloader, testloader = preprocessing_data(f_data, t_data, args)

    # Load model & parameter setting
    model = LstmModel(input_dim=args.features_size, hidden_size=args.hidden_size, output_size=args.output,
                      num_layers=args.num_layers, batch_size=args.batch_size)

    model = model.to(device)

    loss = loss_fn(args.loss_fn)
    optim = optim_fn(model, args.optim, learning_rate=args.lr)

    # Record losses
    epochs = []
    train_losses = []
    val_losses = []

    # Train and validation
    times = 0.0
    for epoch in range(args.epochs):
        ts = time.time()
        model, train_loss = trainer(model, trainloader, optim, loss)
        val_loss = validation(model, valloader, loss)
        te = time.time()
        times += (te - ts)

        train_loss = train_loss / len(trainloader)
        val_loss = val_loss / len(valloader)
        print(f'[{epoch + 1}/{args.epochs}] - {te - ts:.2f}sec, train_loss:{train_loss:.4f}, val_loss: {val_loss:.4f}')

        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Test & calculator evaluation Acc
    y_true, y_pred = tester(model, testloader)

    y_true_invert = tscale.inverse_transform(y_true.reshape(-1, 1))
    y_pred_invert = tscale.inverse_transform(y_pred.reshape(-1, 1))

    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend()
    plt.show()

    DataEval(y_true_invert, y_pred_invert).calc()

    plt.plot(y_true_invert[-144:], label='actual')
    plt.plot(y_pred_invert[-144:], label='predict')
    plt.legend()
    plt.show()
