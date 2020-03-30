from classify_trainer import ClassifyTrainer
from net import DogCatNet
from dog_cat_dataset import DogCatDataset, Normalize, ToTensor
from torchvision import transforms
from auto_gpu import auto_gpu


class DogCatTrainer(ClassifyTrainer):
    def __init__(self):
        super(DogCatTrainer, self).__init__(2, './dog_cat_ckpt')
        self.learning_rate = 1
        self.epoch_size = 20
        self.batch_size = 32
        self.log_step = 20
        self.save_every_epoch = 2

    def set_model(self):
        self.model = auto_gpu(DogCatNet())

    def set_dataset(self):
        train_dir = 'F:\\Data\\training_set\\training_set'
        test_dir = 'F:\\Data\\test_set\\test_set'
        transform = transforms.Compose([Normalize(),
                                       ToTensor()])
        self.train_dataset = DogCatDataset(train_dir, transform, (128, 128))
        self.test_dataset = DogCatDataset(test_dir, transform, (128, 128))

if __name__ == '__main__':
    dog_cat_trainer = DogCatTrainer()
    dog_cat_trainer.train()
