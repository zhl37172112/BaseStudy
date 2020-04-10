from classify_trainer import ClassifyTrainer
from net import DogCatNet
from dog_cat_dataset import DogCatDataset, Normalize, ToTensor, DataAug, Resize
from torchvision import transforms
from auto_gpu import auto_gpu


class DogCatTrainer(ClassifyTrainer):
    def __init__(self):
        super(DogCatTrainer, self).__init__(2, 'dog_cat_ckpt')
        self.learning_rate = 0.01
        self.epoch_size = 30
        self.batch_size = 32
        self.log_step = 20
        self.save_every_epoch = 2

    def set_model(self):
        self.model = auto_gpu(DogCatNet())

    def set_dataset(self):
        train_dir = 'E:\\temp\\cat-and-dog\\training_set\\training_set'
        test_dir = 'E:\\temp\\cat-and-dog\\test_set\\test_set'
        transform_train = transforms.Compose([DataAug(), Resize((256, 256)), Normalize(),
                                              ToTensor()])
        transform_test = transforms.Compose([Resize((256, 256)), Normalize(),
                                             ToTensor()])
        self.train_dataset = DogCatDataset(train_dir, transform_train)
        self.test_dataset = DogCatDataset(test_dir, transform_test)


if __name__ == '__main__':
    dog_cat_trainer = DogCatTrainer()
    dog_cat_trainer.train()
