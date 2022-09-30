#!/usr/bin/env python3
"""
Main training module
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from facial_keypoints_detection.data import FacialKeypointsDataset, Transformer
from facial_keypoints_detection.models import Net

# watch for any changes in model.py, if it changes, re-load it automatically
#  %load_ext autoreload
#  %autoreload 2


class Trainer:
    def __init__(
        self,
        batch_size: int = 10,
        n_epochs: int = 1,
        learning_rate: float = 1e-4,
        use_gpu: bool = True,
    ):
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._use_gpu = use_gpu
        self._net: nn.Module = self.gpu(Net())
        self._criterion = self.gpu(nn.MSELoss())
        self._learning_rate = learning_rate

        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        self._optimizer = optim.Adam(
            params=self._net.parameters(),
            lr=self._learning_rate,
        )

        self._data_transformers = transforms.Compose(
            [
                Transformer.Rescale(250),
                Transformer.RandomCrop(224),
                Transformer.Normalize(),
                Transformer.ToTensor(),
            ]
        )
        self._total_batch_loss: list = []
        self._epoch_loss = []

    @property
    def has_cuda(self):
        return self._use_gpu and torch.cuda.is_available()

    def gpu(self, item: nn.Module):
        if self.has_cuda:
            return item.cuda()
        return item

    def load_data(self, data_path: Path, data_type: str = "training"):
        # create the transformed dataset
        assert self._data_transformers is not None, "No valid data transformers!"

        dataset = FacialKeypointsDataset(
            csv_file=data_path / f"/data/{data_type}_frames_keypoints.csv",
            root_dir=data_path / f"{data_type}",
            transform=self._data_transformers,
        )

        print("Number of {0:10} images: {1:6}".format(data_type, len(dataset)))

        # load training data in batches
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=4,
        )

        return (dataset, loader)

    def reload_all_data(self):
        self._train_dataset, self._train_loader = self.load_data(data_type="training")
        self._test_dataset, self._test_loader = self.load_data(data_type="test")

    # test the model on a batch of test images
    def sample_output(self):
        # iterate through the test dataset
        assert self._test_loader is not None, "No valid test data loader!"

        self._net.to("cpu")

        for i, sample in enumerate(self._test_loader):
            # get sample data: images and ground truth keypoints
            images = sample["image"]
            key_pts = sample["keypoints"]

            # convert images to FloatTensors
            images = images.type(torch.FloatTensor)
            print(f"batch size is {images.shape}.")

            # forward pass to get net output
            output_pts = self._net(images)

            # reshape to batch_size x 68 x 2 pts
            output_pts = output_pts.view(output_pts.size()[0], 68, -1)

            # break after first image is tested
            if i == 0:
                return images, output_pts, key_pts

    def train(self):
        assert self._net is not None
        assert self._train_loader is not None
        assert self._criterion is not None
        assert self._optimizer is not None

        self._net.train()

        total_batch_loss = []
        epoch_loss = []

        for epoch in range(self._n_epochs):
            running_loss = 0.0
            batch_loss = []
            epoch_start_time = time.time()
            # train on batches of data, assumes you already have loader
            for batch_i, data in enumerate(self._train_loader):
                # get the input images and their corresponding labels
                images = data["image"]
                key_pts = data["keypoints"]

                # flatten pts
                key_pts = key_pts.view(key_pts.size(0), -1)

                # convert variables to floats for regression loss
                key_pts = self.gpu(key_pts.type(torch.FloatTensor))
                images = self.gpu(images.type(torch.FloatTensor))

                # forward pass to get outputs
                output_pts = self._net(images)

                # calculate the loss between predicted and target keypoints
                # print(output_pts, key_pts)
                loss = self._criterion(output_pts, key_pts)
                #  print(output_pts)
                #  print(key_pts)

                # zero the parameter (weight) gradients
                self._optimizer.zero_grad()

                # backward pass to calculate the weight gradients
                loss.backward()

                # update the weights
                self._optimizer.step()

                # print loss statistics
                running_loss += loss.item()
                batch_loss.append(loss.item())

                print(
                    f"Epoch: {epoch + 1:3}, Batch: {batch_i + 1:4}, "
                    f"Batch Loss: {loss.item():12.6f}, "
                    f"Elapsed Time: {time.time() - epoch_start_time:15.4f} s."
                )

            total_batch_loss.append(batch_loss)
            epoch_loss.append(running_loss)

        print("Finished Training")
        return (total_batch_loss, epoch_loss)

    def visualize_output(self, test_images, test_outputs, gt_pts=None):
        plt.figure(figsize=(40, 30))

        self._net.to("cpu")
        self._net.eval()

        for i in range(self._batch_size):
            ax = plt.subplot(self._batch_size / 10, 10, i + 1)

            # un-transform the image data
            image = test_images[i].data  # get the image from it's Variable wrapper
            image = image.numpy()  # convert to numpy array from a Tensor
            image = np.transpose(
                image, (1, 2, 0)
            )  # transpose to go from torch to numpy image

            # un-transform the predicted key_pts data
            p_key_pts = test_outputs[i].data
            p_key_pts = p_key_pts.numpy()
            # undo normalization of keypoints
            p_key_pts = p_key_pts * 50.0 + 100

            # plot ground truth points for comparison, if they exist
            ground_truth_pts = None
            if gt_pts is not None:
                ground_truth_pts = gt_pts[i]
                ground_truth_pts = ground_truth_pts * 50.0 + 100

            # call show_all_keypoints
            self.show_all_keypoints(ax, np.squeeze(image), p_key_pts, ground_truth_pts)

            plt.axis("off")

        plt.show()

    def show_all_keypoints(self, ax, image, predicted_key_pts, gt_pts=None):
        """Show image with predicted keypoints"""
        # image is grayscale
        ax.imshow(image, cmap="gray")
        ax.scatter(
            predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=1, marker=".", c="m"
        )
        # plot ground truth points as green pts
        if gt_pts is not None:
            ax.scatter(gt_pts[:, 0], gt_pts[:, 1], s=1, marker=".", c="g")

    def save_model(self):
        model_dir = "saved_models/"
        model_name = f"keypoints_model_{time.time()}.pt"

        # after training, save your model parameters in the dir 'saved_models'
        torch.save(self._net.state_dict(), model_dir + model_name)

    def validate(self):
        self._net.to("cpu")
        self._net.eval()
        images, predicted, gt = self._net_sample_output()
        predicted = predicted.view(predicted.size(0), 68, -1)
        self.visualize_output(images, predicted, gt)

    def plot_batch_loss(self):
        for loss in self._total_batch_loss:
            plt.plot(loss)
        plt.show()

    def plot_epoch_loss(self):
        plt.plot(self._epoch_loss)
        plt.show()

    def train(self):
        print("Start training ...")
        print(f"Number of epochs {self._n_epochs}, batch size {self._batch_size}.")

        if self.has_cuda:
            torch.cuda.empty_cache()

        self.reload_all_data()

        self._total_batch_loss, self._epoch_loss = self.train()
        self.plot_batch_loss()
        self.plot_epoch_loss()
        self.validate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="path to the data set",
    )
    parser.add_argument(
        "--model-output-dir", type=Path, help="output path for the trained model"
    )
    parser.add_argument(
        "--batch-size",
        action="store",
        type=int,
        default=10,
        help="batch size",
    )
    parser.add_argument(
        "--n-epochs",
        action="store",
        type=int,
        default=10,
        help="number of epochs",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="try to use gpu (cuda)",
    )

    options = parser.parse_args()

    trainer = Trainer(
        batch_size=options.batch_size,
        n_epochs=options.n_epochs,
        use_gpu=options.use_gpu,
    )

    trainer.train()


if __name__ == "__main__":
    main()
