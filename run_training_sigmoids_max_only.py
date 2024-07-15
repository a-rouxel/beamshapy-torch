import matplotlib.pyplot as plt
from units import *
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from helpers import load_yaml_config
from simulation import Simulation
from sources import Source
from slm import SLM
from ft_lens import FT_Lens
from FieldGeneration import generate_target_profiles
from torch.utils.tensorboard import SummaryWriter
from cost_functions import *
import torch.nn.functional as F


class FixedDataset(Dataset):
    def __init__(self, source_field, list_target_field):
        self.source_field = source_field
        self.list_target_field = list_target_field

    def __len__(self):
        return 1  # Only one element in the dataset

    def __getitem__(self, idx):
        return self.source_field, self.list_target_field

class OpticalSystem(pl.LightningModule):
    def __init__(self,log_dir="./logs",device="cpu",target_mode_nb=2,try_nb=0):
        super().__init__()
        self.config_source = load_yaml_config("./configs/source.yml")
        self.config_slm = load_yaml_config("./configs/SLM.yml")
        self.config_simulation = load_yaml_config("./configs/simulation.yml")

        self.simulation = Simulation(config_dict=self.config_simulation)
        self.source = Source(config_dict=self.config_source, XY_grid=self.simulation.XY_grid)
        self.ft_lens = FT_Lens(self.simulation.delta_x_in, self.simulation.XY_grid, self.source.wavelength)

        self.target_mode = target_mode_nb

        if self.target_mode %2 == 0:
            self.mode_parity = "even"
        else:
            self.mode_parity = "odd"


        self.list_target_files = generate_target_profiles(yaml_file="./configs/target_profile.yml",
                                                     XY_grid=self.ft_lens.XY_output_grid,
                                                     list_modes_nb=[0,1,2,3,4,5,6,7,8])

        self.target_field = self.list_target_files[self.target_mode]
        self.try_nb = try_nb

        self.best_loss = float('inf')
        inverse_fourier_field = self.ft_lens(self.target_field, pad=False, flag_ifft=True)

        # phase = torch.angle(inverse_fourier_field) + torch.pi / 2
        # # # if abs(phase) <0.01 replace with 0
        # phase[torch.abs(phase) < torch.pi / 2] = 0.000001
        # phase[torch.abs(phase) > torch.pi / 2] = torch.pi - 0.000001

        self.slm = SLM(config_dict=self.config_slm, XY_grid=self.simulation.XY_grid)
        self.slm.amplitude.requires_grad = False

        self.out_field = None
        self.log_dir = log_dir
        self.writer = SummaryWriter(self.log_dir)

        if device == "gpu":
            device = "cuda"
        else:
            device = "cpu"
        self.weights_map = self.hypergaussian_weights(XY_grid=self.ft_lens.XY_output_grid, x_center=0, y_center=0, sigma_x=19*um, sigma_y=19*um, order=12).to(device)
        #
    def print_trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f'Trainable parameter: {name}')

    def forward(self, x):

        modulated_field = self.slm.apply_phase_modulation_sigmoid(x,steepness=2 + self.current_epoch/200,num_terms=1, spacing=1,mode_parity=self.mode_parity)

        self.out_field = self.ft_lens(modulated_field, pad=False, flag_ifft=False)
        return self.out_field

    def hypergaussian_weights(self, XY_grid, x_center, y_center, sigma_x, sigma_y, order):
        # Calculate distances from center
        x_grid = XY_grid[0]  # X coordinates of the grid
        y_grid = XY_grid[1]  # Y coordinates of the grid

        x_dist = (x_grid - x_center) ** 2 / (2 * sigma_x ** 2)
        y_dist = (y_grid - y_center) ** 2 / (2 * sigma_y ** 2)

        # Calculate hypergaussian weights
        weights_x = torch.exp(-x_dist ** order)
        weights_y = torch.exp(-y_dist ** order)

        # Produce a square-shaped weight map by multiplying independent x and y weights
        weight_map = weights_x * weights_y
        # weight_map /= weight_map.sum()  # Normalize the weights to sum to 1
        return weight_map



    def compute_loss(self, out_field, list_target_field):

        energy_out = torch.sum(torch.sum(torch.abs(out_field)**2 ))
        if energy_out == 0:
            energy_out += 1e-10  # Add a small number to avoid divide by zero

        # Apply the weight map to the fields
        weighted_out_field = (out_field * self.weights_map)

        self.inside_energy_percentage = torch.sum(torch.sum(torch.abs(weighted_out_field)**2))/energy_out

        list_overlaps = []

        for idx,target_field in enumerate(list_target_field):

            energy_target = torch.sum(torch.sum(torch.abs(target_field)))
            target_field_norm = target_field  * energy_out/energy_target

            weighted_target_field = (target_field_norm * self.weights_map)
            overlap = calculate_normalized_overlap(weighted_out_field, weighted_target_field)
            list_overlaps.append(overlap)


        list_overlap_tmp = list_overlaps.copy()

        loss = -1 * torch.log(list_overlap_tmp[self.target_mode] + 1e-10)  # Avoid log(0)

        list_overlap_tmp.pop(self.target_mode)


        # loss += -10 * torch.log(1 - torch.max(torch.stack(list_overlap_tmp)))


        # if loss == Nane
        if torch.isnan(loss):
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(self.slm.phase_parameters.detach().numpy())
            ax[1].imshow(self.slm.phase.detach().numpy())
            plt.show()


        # Calculate the differences along each dimension
        diff_dim0 = torch.diff(self.slm.phase, dim=0) ** 2
        diff_dim1 = torch.diff(self.slm.phase, dim=1) ** 2

        # Pad the last row and column of the difference tensors to match the original size
        diff_dim0_padded = F.pad(diff_dim0, (0, 0, 0, 1), 'constant', 0)
        diff_dim1_padded = F.pad(diff_dim1, (0, 1, 0, 0), 'constant', 0)

        # Now you can add the padded tensors together since they match in size
        TV_term = torch.sum(torch.sqrt(diff_dim0_padded + diff_dim1_padded + 1e-2))

        loss += TV_term*1e-5
        # loss +=  -1 * torch.log(self.inside_energy_percentage + 1e-10)

        return loss, list_overlaps


    def training_step(self, batch, batch_idx):
        source_field, list_target_field = batch

        source_field = source_field.clone().detach().requires_grad_(True)  # Clone and detach
        out_field = self.forward(source_field)  # Directly use source field to ensure no input issues

        loss, list_overlap = self.compute_loss(out_field, list_target_field)

        self.current_loss = loss

        # Log gradients
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         self.writer.add_scalar(f'grad_norm/{name}', param.grad.norm(), self.global_step)
        #
        for i, overlap in enumerate(list_overlap):
            self.log(f'list_overlap/mode_{i}', overlap, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log('inside_energy_percentage', self.inside_energy_percentage, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # self.log('source waist', self.source.sigma*2/mm, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def on_after_backward(self):
        # Check for NaNs in all parameters
        for name, param in self.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN detected in gradients of {name} after backward")

    def on_train_epoch_end(self, outputs=None) -> None:


        if self.current_epoch % 100 == 0:

            print("Saving the model")

            # if self.current_loss < self.best_loss:
            #     self.best_loss = self.current_loss
            #     # Save the current best phase
            best_phase = self.slm.phase.cpu().detach().numpy()
            np.save(f'{self.log_dir}/best_slm_phase_{self.target_mode}_{self.try_nb}.npy', best_phase)

            # Continue with your visualization and saving routines
            output_field = torch.abs(self.out_field.cpu().detach()).numpy()
            output_field_norm = (output_field - output_field.min()) / (output_field.max() - output_field.min())
            self.writer.add_image("output_field", output_field_norm, self.current_epoch)

            target_field = torch.abs(self.target_field.unsqueeze(0).cpu().detach()).numpy()
            target_field_norm = (target_field - target_field.min()) / (target_field.max() - target_field.min())
            self.writer.add_image("target_field", target_field_norm, self.current_epoch)

            # Current SLM phase
            slm_phase = self.slm.phase.cpu().detach().unsqueeze(0).numpy()
            slm_phase_norm = (slm_phase - slm_phase.min()) / (slm_phase.max() - slm_phase.min())
            self.writer.add_image("slm_phase", slm_phase_norm, self.current_epoch)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-1)
        return {"optimizer": optimizer}



    def train_dataloader(self):
        dataset = FixedDataset(self.source.field.field, self.list_target_files)
        return DataLoader(dataset, batch_size=1)




from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping



device = "gpu"

for try_nb in range(2):
    for i in range(9):

        model = OpticalSystem(log_dir="./lightning_logs",device=device,target_mode_nb=i,try_nb=try_nb)
        # model.print_trainable_parameters()

        tensorboard_logger = TensorBoardLogger("./lightning_logs", name="my_model")  # Specify the log directory

        early_stop_callback = EarlyStopping(
        monitor='train_loss',
        min_delta=0.00,
        patience=2000,  # Number of epochs with no improvement after which training will be stopped
        verbose=False,
        mode='min'
        )

        trainer = pl.Trainer(
        enable_progress_bar = True, # for turning off progress bar
        enable_model_summary= True, # for turning off weight summary.
        max_epochs          = 12000,
        enable_checkpointing= False,
        profiler            = None,
        logger              = None,
        accelerator=device,
        callbacks=[early_stop_callback],
        )
        trainer.fit(model)
    else:
        pass
