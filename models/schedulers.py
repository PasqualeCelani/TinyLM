
class StepLR:
    def __init__(self, optimizer, step_size, gamma):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.initial_lr = self.optimizer.lr
        self.epoch = 0
    
    def step(self):
        if (self.epoch + 1) % self.step_size == 0:
            new_lr = self.initial_lr * (self.gamma ** ((self.epoch + 1) // self.step_size))
            self.optimizer.lr.assign(new_lr).realize()
        self.epoch += 1