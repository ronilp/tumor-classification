from training_config import BASE_LR, LR_DECAY_EPOCHS, LR_DECAY
from sklearn.metrics import mean_squared_error, r2_score

# This function changes the learning rate over the training model
def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=LR_DECAY_EPOCHS):
    lr = init_lr * (LR_DECAY ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, r2