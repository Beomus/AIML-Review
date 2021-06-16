import torch
import os


device = "cpu"
dtype = torch.float32


def save_checkpoint(filename, model, optimizer, train_acc, epoch):
    save_state = {
        "state_dict": model.state_dict(),
        "acc": train_acc,
        "epoch": epoch + 1,
        "optimizer": optimizer.state_dict(),
    }
    print()
    print("Saving current parameters")
    print("___________________________________________________________")

    torch.save(save_state, filename)


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training or validation set")
    else:
        print("Checking accuracy on test set")
    num_correct = 0
    num_samples = 0
    # model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = (float(num_correct) / num_samples) * 100.0
        print("Got %d / %d correct (%.2f)" % (num_correct, num_samples, acc))
        return acc


def load_model(args, model, optimizer):
    if args.resume:
        model.eval()
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"]
            best_acc = checkpoint["acc"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            return model, optimizer, checkpoint, start_epoch, best_acc
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print("No pretrained model. Starting from scratch!")
