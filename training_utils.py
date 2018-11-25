from classification_config import BASE_LR, LR_DECAY_EPOCHS, LR_DECAY

# This function changes the learning rate over the training model
def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=LR_DECAY_EPOCHS):
    lr = init_lr * (LR_DECAY ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer