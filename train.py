import argparse

from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import ModuleList, Parameter
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
from torchmetrics import Accuracy
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from classifier import Classifier
from doe import DOE
from propagate import Propagate
from spatial_coherence import get_exponentially_decaying_spatial_coherence, get_source_modes

import time

torch.set_float32_matmul_precision('medium') #precision for L4

class DiffractiveSystem(pl.LightningModule):
    def __init__(self, learning_rate, gamma, coherence_degree, wavelength, pixel_size):
        super().__init__()
        self.device_param = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # 记录当前设备
        self.save_hyperparameters()
        self.doe_list = ModuleList([DOE(shape=100) for _ in range(4)])
        self.initial_propagate = Propagate(
            preceding_shape=28 * 4,
            succeeding_shape=100,
            propagation_distance=0.05,
            wavelength=wavelength,
            pixel_size=pixel_size,
            device=self.device_param,
        )
        self.intralayer_propagate = Propagate(
            preceding_shape=100,
            succeeding_shape=100,
            propagation_distance=0.05,
            wavelength=wavelength,
            pixel_size=pixel_size,
            device=self.device_param,
        )
        self.last_propagate = Propagate(
            preceding_shape=100,
            succeeding_shape=50,
            propagation_distance=0.05,
            wavelength=wavelength,
            pixel_size=pixel_size,
            device=self.device_param,
        )
        self.first_propagate = Propagate(
            preceding_shape=50,
            succeeding_shape=100,
            propagation_distance=0.05,
            wavelength=wavelength,
            pixel_size=pixel_size,
            device=self.device_param,
        )
        self.classifier = Classifier(shape=50, region_size=12)
        self.source_modes = get_source_modes(shape=28, image_pixel_size=4)
        self.accuracy = Accuracy("multiclass", num_classes=10)

        self.alpha_s = (torch.tensor(0.8))
        self.alpha_ns = (torch.tensor(0.1))
        self.I_sat = Parameter((torch.tensor(1) * 1e-6))

    def saturable_absorption(self, x: torch.Tensor) -> torch.Tensor:
        '''x is the field'''
        T = (1 - self.alpha_s / (1 + torch.abs(x)**2 / self.I_sat - self.alpha_ns))
        amplitude_modulation = torch.sqrt(T)
        return amplitude_modulation * x

    def forward(self, x):
        #x: (batch_size, 1, 28, 28) -> return: (batch_size, 10)
        coherence_tensor = get_exponentially_decaying_spatial_coherence(
            torch.squeeze(x, -3).to(torch.cdouble), self.hparams.coherence_degree
        ) #(batch_size, 28, 28, 28, 28)

        modes = self.source_modes # (28*28, 28*4, 28*4)
        modes = self.initial_propagate(modes) # (28*28, 100, 100)
        
        for doe in self.doe_list:
            modes = doe(modes)
            modes = self.intralayer_propagate(modes)
        #(28*28, 100, 100)

        modes = self.last_propagate(modes)       

        batch_size = coherence_tensor.shape[0]
        total_input_pixels = coherence_tensor.shape[-2] * coherence_tensor.shape[-1]
        total_output_pixels = modes.shape[-2] * modes.shape[-1]

        
        # output_intensity = (
        #     torch.einsum(  # Reduce precision to cfloat for performance
        #         "bij, io, jo-> bo",
        #         coherence_tensor.view(batch_size, total_input_pixels, total_input_pixels).to(torch.cfloat),
        #         modes.view(total_input_pixels, total_output_pixels).conj().to(torch.cfloat),
        #         modes.view(total_input_pixels, total_output_pixels).to(torch.cfloat),
        #     )
        #     .real.view(batch_size, *modes.shape[-2:]) #(batch_size, 100, 100)
        #     .to(torch.double)
        # )

        field_5 = (
            torch.einsum(  # Reduce precision to cfloat for performance
                "bij, io, jp-> bop",
                coherence_tensor.view(batch_size, total_input_pixels, total_input_pixels).to(torch.cfloat),
                modes.view(total_input_pixels, total_output_pixels).conj().to(torch.cfloat),
                modes.view(total_input_pixels, total_output_pixels).to(torch.cfloat),
            )
            .to(torch.cdouble)
        ) # (batch_size, 50*50, 50*50)

        output_sa = self.saturable_absorption(field_5)

        # modes = get_source_modes(shape=50, image_pixel_size=2).to(self.device_param)
        # # modes = self.first_propagate(modes)
        # for doe in self.doe_list:
        #     modes = doe(modes)
        #     modes = self.intralayer_propagate(modes)
        # modes = self.last_propagate(modes)
        # # print("modes",modes.device)

        # total_input_pixels = output_sa.shape[-1]
        # total_output_pixels = modes.shape[-2] * modes.shape[-1]

        # output = (
        #     torch.einsum(  # Reduce precision to cfloat for performance
        #         "bij, io, jp-> bop",
        #         output_sa.view(batch_size, total_input_pixels, total_input_pixels).to(torch.cfloat),
        #         modes.view(total_input_pixels, total_output_pixels).conj().to(torch.cfloat),
        #         modes.view(total_input_pixels, total_output_pixels).to(torch.cfloat),
        #     )
        #     .to(torch.double)
        # ) # (batch_size, 50*50, 50*50)

        output_intensity = output_sa.diagonal(dim1=-2, dim2=-1).real.view(batch_size, *modes.shape[-2:]) #(batch_size, 50*50)

        # coherence_tensor = coherence_tensor.view(batch_size, total_input_pixels, total_input_pixels).to(torch.cfloat) #bij
        # modes_i = modes.view(total_input_pixels, total_output_pixels).conj().to(torch.cfloat) #io
        # modes_j = modes.view(total_input_pixels, total_output_pixels).to(torch.cfloat) #jo

        # intermediate = torch.matmul(coherence_tensor, modes_j)  #bij * jo -> bio
        # # 结果 shape: (batch_size, total_input_pixels, total_output_pixels)
        # temp = modes_i.unsqueeze(0).expand(intermediate.shape[0], -1, -1) #io -> bio

        # output = torch.sum(intermediate * temp, dim=1)  #bio * bio -> bo
        # # 结果 shape: (batch_size, total_output_pixels)

        # output_intensity = output.real.view(batch_size, *modes.shape[-2:]).to(torch.double)
        
        return self.classifier(output_intensity)

    def training_step(self, batch, batch_idx):
        # start_time = time.time()  # 记录起始时间
        data, target = batch
        output = self(data)
        loss = cross_entropy(output, target)
        # end_time = time.time()  # 记录结束时间
        # print(f"运行时间: {end_time - start_time:.6f} 秒")

        
        # torch.cuda.synchronize()  # 确保 GPU 任务完成
        # start_time = time.time()  # 记录起始时间
        

        acc = self.accuracy(output, target)
        # print("Output device:", output.device)
        # print("Target device:", target.device)
        # print(self.accuracy.device)
        # start_time = time.time()  # 记录起始时间
        # torch.cuda.synchronize()  # 确保 GPU 任务完成

        # end_time = time.time()  # 记录结束时间
        # print(f"运行时间: {end_time - start_time:.6f} 秒")
        
        # start_time = time.time()  # 记录起始时间
        self.log("train_loss", loss, sync_dist=False)
        self.log("train_acc", acc, sync_dist=False)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"], sync_dist=False)
        # end_time = time.time()  # 记录结束时间
        # print(f"运行时间: {end_time - start_time:.6f} 秒")
        
        return loss

    def validation_step(self, batch, batch_idx):
        # start_time = time.time()  # 记录起始时间
        data, target = batch
        output = self(data)
        loss = cross_entropy(output, target)
        acc = self.accuracy(output, target)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        # end_time = time.time()  # 记录结束时间
        # print(f"运行时间: {end_time - start_time:.6f} 秒")

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = cross_entropy(output, target)
        acc = self.accuracy(output, target)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

class MyCallback(Callback):
  def on_validation_epoch_end(self, trainer, pl_module):
    val_acc = trainer.callback_metrics.get("val_acc")
    if val_acc is not None:
        print(f"Epoch {trainer.current_epoch} - val_acc: {val_acc:.4f}")

    if hasattr(pl_module, 'last_T'):
      T = pl_module.last_T
      # 将 T 转为 numpy 数组，并展开为一维
      T_np = T.cpu().numpy().flatten()
      
      # 绘制直方图
      plt.figure()
      plt.hist(T_np, bins=50)
      plt.title(f"Epoch {trainer.current_epoch} - T Distribution")
      plt.xlabel("T value")
      plt.ylabel("Frequency")
      # plt.show(block=False)

      filename = f"./plot/t_distribution_epoch_{trainer.current_epoch}.png"
      plt.savefig(filename)
      plt.close()
      print(f"Saved T distribution histogram to {filename}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 选择更快的本地存储
    DATA_DIR = "/tmp/mnist_data"

    torch.manual_seed(args.seed)
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]

    transform = transforms.Compose(transform_list)
    full_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    dataset = Subset(full_dataset, indices=range(10000,20000)) # subset 10000
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True,prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True,prefetch_factor=2, persistent_workers=True)
    test_loader = DataLoader(
        datasets.MNIST(DATA_DIR, train=False, transform=transform),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2, 
        persistent_workers=True
    )

    model = DiffractiveSystem(args.lr, args.gamma, args.coherence_degree, args.wavelength, args.pixel_size)

    my_callback = MyCallback()
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints_SA",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="model-{epoch:02d}-{val_acc:.2f}",
        verbose=True,
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(precision=16, enable_checkpointing=True, max_epochs=args.epochs, callbacks=[checkpoint_callback, my_callback], accelerator=accelerator)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--coherence-degree", type=float, required=True, help="coherence degree")
    parser.add_argument("--wavelength", type=float, default=700e-9, help="field wavelength (default: 700 nm)")
    parser.add_argument("--pixel-size", type=float, default=10e-6, help="field pixel size (default: 10 um)")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="input batch size for training (default: 32)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs to train (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate (default: 1e-2)")
    parser.add_argument("--num-workers", type=int, default=1, help="number of workers (default: 1)")
    parser.add_argument("--gamma", type=float, default=0.95, help="Learning rate step gamma (default: 0.95)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    args = parser.parse_args()
    main(args)
